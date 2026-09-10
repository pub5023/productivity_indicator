# -*- coding: utf-8 -*-
"""
Unified code (DISPLAYMATCH version)
- Load raw CSV
- Normalize inconsistencies in occupation_id / occupation_name (e.g., id=3, 8, 8.2 and variations like 8-2)
- Estimate task-level ground truth (summary / instruction / presentation) using BM × time via least squares, then clip to [0,1]
- Reconstruct task-level estimated reduction rates from importance-weighted W and compute task_error = est - task_truth
- Run three regressions (HC3) and save the rounded DataFrame for display directly to CSV (DISPLAYMATCH)
"""

import os
import re
import hashlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# === Paths ===
PATH_TIME_W = "../0.input_data/Time_Weighted_Average_Data_W.csv"
PATH_TIME_L = "../0.input_data/Time_Weighted_Average_Data_L.csv"
PATH_IMP_W  = "../0.input_data/Importance_Weighted_Average_Data_W.csv"
PATH_IMP_L  = "../0.input_data/Importance_Weighted_Average_Data_L.csv"
PATH_PROXY  = "./in_task_truth_regression_fullcode/proxy_model_summary.csv"  # Optional (works even if not provided)
PATH_LONG  = "./in_task_truth_regression_fullcode/proxy_by_file_long.csv"  # Optional (works even if not provided)

# === Outputs (DISPLAYMATCH) ===
OUT_SEG      = "./out_task_truth_regression_fullcode/TASKTRUTH_reg_segment_task.csv"
OUT_IMP      = "./out_task_truth_regression_fullcode/TASKTRUTH_reg_importance_task_segment.csv"
OUT_PROXY    = "./out_task_truth_regression_fullcode/TASKTRUTH_reg_proxy_task_segment.csv"
OUT_MODEL    = "./out_task_truth_regression_fullcode/TASKTRUTH_model_level_errors.csv"
OUT_TASKTRUE = "./out_task_truth_regression_fullcode/TASKTRUTH_task_truth_BM_time_clipped.csv"
OUT_MD5      = "./out_task_truth_regression_fullcode/TASKTRUTH_inputs_md5.csv"
OUT_DEFICIT  = "./out_task_truth_regression_fullcode/proxy_deficit.csv"

# =========================================================
# 1) Load raw data
# =========================================================
timeW0 = pd.read_csv(PATH_TIME_W)
timeL0 = pd.read_csv(PATH_TIME_L)
impW0  = pd.read_csv(PATH_IMP_W)
impL0  = pd.read_csv(PATH_IMP_L)


# =========================================================
# 2) Normalize occupation_id + unify occupation_name (id=3/8/8.2)
# =========================================================
def normalize_occ_id(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

# Normalize notation: convert "8-2" variations to "8.2"
ID_STANDARD_MAP = {"8-2": "8.2", "８-２": "8.2", "８.２": "8.2"}
TARGET_IDS = {"3", "8", "8.2"}

def standardize_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["occupation_id"] = df["occupation_id"].apply(normalize_occ_id).replace(ID_STANDARD_MAP)
    return df

timeW = standardize_ids(timeW0)
timeL = standardize_ids(timeL0)
impW  = standardize_ids(impW0)
impL  = standardize_ids(impL0)

# Create canonical occupation_name (use the most frequent one across all four files)
all_occ = pd.concat([
    timeW[["occupation_id", "occupation_name"]],
    timeL[["occupation_id", "occupation_name"]],
    impW[["occupation_id", "occupation_name"]],
    impL[["occupation_id", "occupation_name"]],
], ignore_index=True)

canon = (
    all_occ[all_occ["occupation_id"].isin(TARGET_IDS)]
    .groupby(["occupation_id", "occupation_name"]).size()
    .reset_index(name="n")
    .sort_values(["occupation_id", "n"], ascending=[True, False])
    .groupby("occupation_id").head(1)
    .reset_index(drop=True)
)

CANON_MAP = dict(zip(canon["occupation_id"], canon["occupation_name"]))

def apply_canon_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["occupation_name"] = df["occupation_id"].map(CANON_MAP).fillna(df["occupation_name"])
    return df

timeW = apply_canon_names(timeW)
timeL = apply_canon_names(timeL)
impW  = apply_canon_names(impW)
impL  = apply_canon_names(impL)


# =========================================================
# 3) Estimate task-level ground truth for BM × time using least squares (clipped)
#    Assume: job-level ground truth (BM) ≈ time_weight_summary*T_s + ...
# =========================================================
occ_time = (
    timeW.groupby(["occupation_id", "occupation_name"], as_index=False)
    .agg(
        task_weight_summary=("task_weight_summary", "first"),
        task_weight_instruction=("task_weight_instruction", "first"),
        task_weight_presentation=("task_weight_presentation", "first"),
        true_bm=("true_reduction_bm", "first"),
    )
)

W_time = occ_time[["task_weight_summary", "task_weight_instruction", "task_weight_presentation"]].to_numpy(float)
T_bm = occ_time["true_bm"].to_numpy(float)

t_hat, residuals, rank, svals = np.linalg.lstsq(W_time, T_bm, rcond=None)
t_hat = t_hat.flatten()
t_clip = np.clip(t_hat, 0, 1)

T_TASK = {
    "summary": float(t_clip[0]),
    "instruction": float(t_clip[1]),
    "presentation": float(t_clip[2]),
}

pd.DataFrame([{
    "baseline": "BM",
    "weights": "time",
    "summary": T_TASK["summary"],
    "instruction": T_TASK["instruction"],
    "presentation": T_TASK["presentation"],
    "rank": int(rank),
}]).to_csv(OUT_TASKTRUE, index=False, encoding="utf-8-sig")


# =========================================================
# 4) Reconstruct task-level estimated reduction rates from importance-weighted W and create task_df
#    task_weight_* represents importance (0–1)
# =========================================================
for t in ["summary", "instruction", "presentation"]:
    impW[f"est_red_{t}"] = 1 - impW[f"est_ai_time_{t}"] / impW[f"est_human_time_{t}"]

task_df = pd.concat([
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "occupation_name": impW["occupation_name"],
        "task": "summary",
        "importance": impW["task_weight_summary"].astype(float),
        "est_reduction": impW["est_red_summary"].astype(float),
    }),
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "occupation_name": impW["occupation_name"],
        "task": "instruction",
        "importance": impW["task_weight_instruction"].astype(float),
        "est_reduction": impW["est_red_instruction"].astype(float),
    }),
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "occupation_name": impW["occupation_name"],
        "task": "presentation",
        "importance": impW["task_weight_presentation"].astype(float),
        "est_reduction": impW["est_red_presentation"].astype(float),
    }),
], ignore_index=True)

task_df["true_task"] = task_df["task"].map(T_TASK).astype(float)
task_df["task_error"] = task_df["est_reduction"] - task_df["true_task"]
task_df["abs_task_error"] = task_df["task_error"].abs()

# Segment definition (fixed)
def assign_segment(model: str) -> str:
    s = str(model).lower()
    if s in ["gpt52p", "gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i", "gpt52t", "opus", "sonnet", "haiku"]):
        return "A"
    return "B"

task_df["segment"] = task_df["model"].apply(assign_segment)

# Add proxy_total (optional)
if os.path.exists(PATH_PROXY):
    proxy = pd.read_csv(PATH_PROXY)[["model", "proxy_total"]].copy()
    task_df["model"] = task_df["model"].str.lower()
    task_df = task_df.merge(proxy, on="model", how="left")
else:
    task_df["proxy_total"] = np.nan


# =========================================================
# 5) Regression (HC3) → round for display (4 decimals) → save directly to CSV (DISPLAYMATCH)
# =========================================================
def coef_table(res, decimals=4):
    out = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "p": res.pvalues.values,
    })
    out["coef"] = out["coef"].round(decimals)
    out["p"] = out["p"].round(decimals)
    return out

# (1) segment + task fixed effects
res_seg = smf.ols("task_error ~ C(segment) + C(task)", data=task_df).fit(cov_type="HC3")
coef_seg = coef_table(res_seg, 4)
coef_seg.to_csv(OUT_SEG, index=False, encoding="utf-8-sig")

# (2) importance*task + segment
res_imp = smf.ols("task_error ~ importance * C(task) + C(segment)", data=task_df).fit(cov_type="HC3")
coef_imp = coef_table(res_imp, 4)
coef_imp.to_csv(OUT_IMP, index=False, encoding="utf-8-sig")

# (3) proxy_total + task + segment (drop rows with missing proxy_total)
if task_df["proxy_total"].notna().any():
    res_proxy = smf.ols(
        "task_error ~ proxy_total + C(task) + C(segment)",
        data=task_df.dropna(subset=["proxy_total"])
    ).fit(cov_type="HC3")
    coef_proxy = coef_table(res_proxy, 4)
else:
    coef_proxy = pd.DataFrame({"term": [], "coef": [], "p": []})

coef_proxy.to_csv(OUT_PROXY, index=False, encoding="utf-8-sig")

# Model-level errors (rounded to 6 decimals: fixed for summary table)
model_mae = (
    task_df.groupby("model", as_index=False)
    .agg(
        task_mae=("abs_task_error", "mean"),
        task_bias=("task_error", "mean"),
        proxy_total=("proxy_total", "first"),
        segment=("segment", "first"),
    )
)
model_mae["task_mae"] = model_mae["task_mae"].round(6)
model_mae["task_bias"] = model_mae["task_bias"].round(6)
model_mae.to_csv(OUT_MODEL, index=False, encoding="utf-8-sig")


# =========================================================
# 6) Input MD5 (for reproducibility)
# =========================================================
def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

md5_tbl = pd.DataFrame([
    {"file": os.path.basename(PATH_TIME_W), "md5": md5_of(PATH_TIME_W)},
    {"file": os.path.basename(PATH_IMP_W),  "md5": md5_of(PATH_IMP_W)},
    {"file": os.path.basename(PATH_PROXY),  "md5": md5_of(PATH_PROXY)} if os.path.exists(PATH_PROXY) else {"file":"proxy_model_summary.csv","md5":""},
])
md5_tbl.to_csv(OUT_MD5, index=False, encoding="utf-8-sig")

# =========================================================
# 7) Missing proxy
# =========================================================
long_df = pd.read_csv(PATH_LONG)
cols = ["residual_per_1k","scope_per_1k","caution_per_1k"]
long_df["proxy_missing_cnt"] = (long_df[cols] == 0).all(axis=1)
long_df = long_df.groupby("model").agg({"proxy_missing_cnt":"sum"})

df_deficit = model_mae[["model", "segment", "proxy_total"]].copy()
df_deficit = df_deficit.merge(long_df, on="model", how="left")
df_deficit.to_csv(OUT_DEFICIT, index=False, encoding="utf-8-sig")

print("Saved:")
print(" ", OUT_TASKTRUE)
print(" ", OUT_SEG)
print(" ", OUT_IMP)
print(" ", OUT_PROXY)
print(" ", OUT_MODEL)
print(" ", OUT_MD5)
print(" ", OUT_DEFICIT)