import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load required datasets (fresh session)
impW = pd.read_csv("../0.input_data/Importance_Weighted_Average_Data_W.csv")
timeW = pd.read_csv("../0.input_data/Time_Weighted_Average_Data_W.csv")
proxy_model = pd.read_csv("./in_appendix_job_base/proxy_model_summary.csv")[["model","proxy_total"]].copy()

# Occupation truth by id
occ_truth = (timeW.groupby("occupation_id", as_index=False)
             .agg(true_bm=("true_reduction_bm","first"),
                  true_ref4=("true_reduction_ref4","first"),
                  occupation_name=("occupation_name","first")))

# Reconstruct task-level estimated reductions from times
for t in ["summary","instruction","presentation"]:
    impW[f"est_red_{t}"] = 1 - impW[f"est_ai_time_{t}"] / impW[f"est_human_time_{t}"]

task_occ = pd.concat([
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "task": "summary",
        "importance": impW["task_weight_summary"].astype(float),
        "est_reduction": impW["est_red_summary"].astype(float),
    }),
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "task": "instruction",
        "importance": impW["task_weight_instruction"].astype(float),
        "est_reduction": impW["est_red_instruction"].astype(float),
    }),
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "task": "presentation",
        "importance": impW["task_weight_presentation"].astype(float),
        "est_reduction": impW["est_red_presentation"].astype(float),
    }),
], ignore_index=True)

task_occ = task_occ.merge(occ_truth[["occupation_id","true_bm","true_ref4","occupation_name"]], on="occupation_id", how="left")

def assign_segment(model: str) -> str:
    s=str(model).lower()
    if s in ["gpt52p","gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i","gpt52t","opus","sonnet","haiku"]):
        return "A"
    return "B"
task_occ["segment"] = task_occ["model"].apply(assign_segment)
task_occ["model"] = task_occ["model"].str.lower()
task_occ = task_occ.merge(proxy_model, on="model", how="left")

task_occ["occ_error_bm"] = task_occ["est_reduction"] - task_occ["true_bm"]
task_occ["occ_abs_error_bm"] = task_occ["occ_error_bm"].abs()

def coef_table(res, decimals=4):
    out = pd.DataFrame({"term": res.params.index, "coef": res.params.values, "p": res.pvalues.values})
    out["coef"] = out["coef"].round(decimals)
    out["p"] = out["p"].round(decimals)
    return out

res_seg = smf.ols("occ_error_bm ~ C(segment) + C(task)", data=task_occ).fit(cov_type="HC3")
tbl_seg = coef_table(res_seg, 4)

res_imp = smf.ols("occ_error_bm ~ importance * C(task) + C(segment)", data=task_occ).fit(cov_type="HC3")
tbl_imp = coef_table(res_imp, 4)

res_proxy = smf.ols("occ_error_bm ~ proxy_total + C(task) + C(segment)",
                    data=task_occ.dropna(subset=["proxy_total"])).fit(cov_type="HC3")
tbl_proxy = coef_table(res_proxy, 4)

model_occ = (task_occ.groupby("model", as_index=False)
             .agg(occ_mae_bm=("occ_abs_error_bm","mean"),
                  occ_bias_bm=("occ_error_bm","mean"),
                  proxy_total=("proxy_total","first"),
                  segment=("segment","first")))
model_occ["occ_mae_bm"] = model_occ["occ_mae_bm"].round(6)
model_occ["occ_bias_bm"] = model_occ["occ_bias_bm"].round(6)

out_seg = "./out_appendix_job_base/APPX_OCCBM_reg_segment_task.csv"
out_imp = "./out_appendix_job_base/APPX_OCCBM_reg_importance_task_segment.csv"
out_proxy = "./out_appendix_job_base/APPX_OCCBM_reg_proxy_task_segment.csv"
out_model = "./out_appendix_job_base/APPX_OCCBM_model_level_errors.csv"

tbl_seg.to_csv(out_seg, index=False, encoding="utf-8-sig")
tbl_imp.to_csv(out_imp, index=False, encoding="utf-8-sig")
tbl_proxy.to_csv(out_proxy, index=False, encoding="utf-8-sig")
model_occ.to_csv(out_model, index=False, encoding="utf-8-sig")

(out_seg, out_imp, out_proxy, out_model)