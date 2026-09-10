import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import re

# ----------------------------
# Create task_error based on task-level ground truth (BM × time),
# and conduct sensitivity analysis with three treatments for missing proxy_total
# ----------------------------

# input
PATH_TASK_TRUTH = "./in_proxy_regression2/TASKTRUTH_task_truth_BM_time_clipped.csv"   # Task-level ground truth for summary / instruction / presentation
PATH_IMP_W      = "../0.input_data/Importance_Weighted_Average_Data_W.csv"                  # Model × occupation (estimated time, importance)
PATH_PROXY_CC   = "./in_proxy_regression2/proxy_model_summary_cc.csv"                    # model, proxy_total
PATH_PROXY_M0   = "./in_proxy_regression2/proxy_model_summary.csv"                    # model, proxy_total
PATH_PROXY_MS   = "./in_proxy_regression2/proxy_model_summary_segmean.csv"                # model, proxy_total
PATH_LONG       = "./in_proxy_regression2/proxy_by_file_long.csv"

# output
OUT_CC = "./out_proxy_regression2/TASKTRUTH_proxy_reg_complete_case.csv"
OUT_M0 = "./out_proxy_regression2/TASKTRUTH_proxy_reg_allmodels_proxy0.csv"
OUT_MS = "./out_proxy_regression2/TASKTRUTH_proxy_reg_allmodels_segmean.csv"
OUT_COMP = "./out_proxy_regression2/proxy_sensitivity_comparison.csv"

# --- Load ---
truth = pd.read_csv(PATH_TASK_TRUTH)
impW  = pd.read_csv(PATH_IMP_W)
proxy_model_cc = pd.read_csv(PATH_PROXY_CC)[["model", "proxy_total"]].copy()
proxy_model_m0 = pd.read_csv(PATH_PROXY_M0)[["model", "proxy_total"]].copy()
proxy_model_ms = pd.read_csv(PATH_PROXY_MS)[["model", "proxy_total"]].copy()
df_long = pd.read_csv(PATH_LONG)

# --- Proxy missing ---
df_long["occupation_id"] = df_long["file"].apply(lambda x: re.search(r"比較_(\d+(?:\.\d+)?)", x).group(1)).astype(float)
df_long["proxy_missing"] = (df_long[["residual_per_1k", "scope_per_1k", "caution_per_1k"]]==0).all(axis=1).astype(int)
impW["model"] = impW["model"].str.lower()
impW = impW.merge(df_long[["model", "occupation_id", "proxy_missing"]], on=["model", "occupation_id"], how="left")

# --- Task truths（BM×time clipped） ---
T_task = {
    "summary": float(truth.loc[0, "summary"]),
    "instruction": float(truth.loc[0, "instruction"]),
    "presentation": float(truth.loc[0, "presentation"]),
}

# --- Reconstruct task-level estimated reductions from times ---
for t in ["summary", "instruction", "presentation"]:
    impW[f"est_red_{t}"] = 1 - impW[f"est_ai_time_{t}"] / impW[f"est_human_time_{t}"]

# --- Build long task-level dataframe ---
task_df = pd.concat(
    [
        pd.DataFrame(
            {
                "model": impW["model"],
                "task": "summary",
                "importance": impW["task_weight_summary"].astype(float),
                "est_reduction": impW["est_red_summary"].astype(float),
                "proxy_missing": impW["proxy_missing"]
            }
        ),
        pd.DataFrame(
            {
                "model": impW["model"],
                "task": "instruction",
                "importance": impW["task_weight_instruction"].astype(float),
                "est_reduction": impW["est_red_instruction"].astype(float),
                "proxy_missing": impW["proxy_missing"]
            }
        ),
        pd.DataFrame(
            {
                "model": impW["model"],
                "task": "presentation",
                "importance": impW["task_weight_presentation"].astype(float),
                "est_reduction": impW["est_red_presentation"].astype(float),
                "proxy_missing": impW["proxy_missing"]
            }
        ),
    ],
    ignore_index=True,
)

# --- Task-truth-based error ---
task_df["true_task"] = task_df["task"].map(T_task).astype(float)
task_df["task_error"] = task_df["est_reduction"] - task_df["true_task"]

# --- Segment assignment（A/B/C） ---
def assign_segment(m: str) -> str:
    s = str(m).lower()
    if s in ["gpt52p", "gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i", "gpt52t", "opus", "sonnet", "haiku"]):
        return "A"
    return "B"

task_df["segment"] = task_df["model"].apply(assign_segment)

# --- Attach proxy_total ---
task_df["model"] = task_df["model"].str.lower()
task_df = task_df.merge(proxy_model_cc, on="model", how="left")

# --- Fill strategies ---
# Strategy 1: proxy_total missing -> 0, with missing indicator
task_df = task_df.merge(proxy_model_m0.rename(columns={"proxy_total":"proxy0"}), on="model")

# Strategy 2: proxy_total missing -> segment mean of observed, with missing indicator
task_df = task_df.merge(proxy_model_ms.rename(columns={"proxy_total":"proxy_segmean"}), on="model")

# ----------------------------
# Regressions (HC3)
# ----------------------------

# A) Complete-case (rows with proxy_total only)
res_cc = smf.ols(
    "task_error ~ proxy_total + C(task) + C(segment)",
    data=task_df.dropna(subset=["proxy_total"]),
).fit(cov_type="HC3")

tbl_cc = pd.DataFrame(
    {"term": res_cc.params.index, "coef": res_cc.params.values, "p": res_cc.pvalues.values}
)
tbl_cc["coef"] = tbl_cc["coef"].round(4)
tbl_cc["p"] = tbl_cc["p"].round(4)
tbl_cc.to_csv(OUT_CC, index=False, encoding="utf-8-sig")

# B) All models: proxy0 + missing indicator
res_m0 = smf.ols(
    "task_error ~ proxy0 + proxy_missing + C(task) + C(segment)",
    data=task_df,
).fit(cov_type="HC3")

tbl_m0 = pd.DataFrame(
    {"term": res_m0.params.index, "coef": res_m0.params.values, "p": res_m0.pvalues.values}
)
tbl_m0["coef"] = tbl_m0["coef"].round(4)
tbl_m0["p"] = tbl_m0["p"].round(4)
tbl_m0.to_csv(OUT_M0, index=False, encoding="utf-8-sig")

# C) All models: proxy_segmean + missing indicator
res_ms = smf.ols(
    "task_error ~ proxy_segmean + proxy_missing + C(task) + C(segment)",
    data=task_df,
).fit(cov_type="HC3")

tbl_ms = pd.DataFrame(
    {"term": res_ms.params.index, "coef": res_ms.params.values, "p": res_ms.pvalues.values}
)
tbl_ms["coef"] = tbl_ms["coef"].round(4)
tbl_ms["p"] = tbl_ms["p"].round(4)
tbl_ms.to_csv(OUT_MS, index=False, encoding="utf-8-sig")

# D) Comparison
cc = tbl_cc[tbl_cc["term"]=="proxy_total"]
p0 = tbl_m0[tbl_m0["term"]=="proxy0"]
p0_missing = tbl_m0[tbl_m0["term"]=="proxy_missing"]
ps = tbl_ms[tbl_ms["term"]=="proxy_segmean"]
ps_missing = tbl_ms[tbl_ms["term"]=="proxy_missing"]

df_out = pd.DataFrame(
    {
        "spec": ["Complete-case", "All models: proxy0 + missing", "All models: seg-mean + missing"],
        "proxy_term": ["proxy_total","proxy0","proxy_segmean"],
        "proxy_coef":[cc["coef"].tolist()[0], p0["coef"].tolist()[0], ps["coef"].tolist()[0]],
        "proxy_p":[cc["p"].tolist()[0], p0["p"].tolist()[0], ps["p"].tolist()[0]],
        "missing_coef":[None, p0_missing["coef"].tolist()[0], ps_missing["coef"].tolist()[0]],
        "missing_p":[None, p0_missing["p"].tolist()[0], ps_missing["p"].tolist()[0]],
    }
)

df_out.to_csv(OUT_COMP, index=False, encoding="utf-8-sig")

print("Saved:")
print(" ", OUT_CC)
print(" ", OUT_M0)
print(" ", OUT_MS)
print(" ", OUT_COMP)