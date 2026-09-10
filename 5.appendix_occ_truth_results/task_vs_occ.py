import pandas as pd, numpy as np
import statsmodels.formula.api as smf

# Load current inputs
impW = pd.read_csv("../0.input_data/Importance_Weighted_Average_Data_W.csv")
timeW = pd.read_csv("../0.input_data/Time_Weighted_Average_Data_W.csv")

# Build occupation truth (BM)
occ_truth = (timeW.groupby("occupation_id", as_index=False)
             .agg(true_bm=("true_reduction_bm","first")))

# Reconstruct est reduction per task from times
for t in ["summary","instruction","presentation"]:
    impW[f"est_red_{t}"] = 1 - impW[f"est_ai_time_{t}"] / impW[f"est_human_time_{t}"]

task_occ = pd.concat([
    pd.DataFrame({"model":impW["model"],"occupation_id":impW["occupation_id"],"task":"summary",
                  "importance":impW["task_weight_summary"].astype(float),"est_reduction":impW["est_red_summary"].astype(float)}),
    pd.DataFrame({"model":impW["model"],"occupation_id":impW["occupation_id"],"task":"instruction",
                  "importance":impW["task_weight_instruction"].astype(float),"est_reduction":impW["est_red_instruction"].astype(float)}),
    pd.DataFrame({"model":impW["model"],"occupation_id":impW["occupation_id"],"task":"presentation",
                  "importance":impW["task_weight_presentation"].astype(float),"est_reduction":impW["est_red_presentation"].astype(float)}),
], ignore_index=True)

# Segment mapping
def assign_segment(model: str) -> str:
    s=str(model).lower()
    if s in ["gpt52p","gpt51t"]: return "C"
    if any(k in s for k in ["gpt52i","gpt52t","opus","sonnet","haiku"]): return "A"
    return "B"
task_occ["segment"]=task_occ["model"].apply(assign_segment)

task_occ = task_occ.merge(occ_truth, on="occupation_id", how="left")
task_occ["occ_error_bm"] = task_occ["est_reduction"] - task_occ["true_bm"]

# Occupation-truth BM regression with importance interaction
res_occ_imp = smf.ols("occ_error_bm ~ importance * C(task) + C(segment)", data=task_occ).fit(cov_type="HC3")
occ_imp = pd.DataFrame({"term":res_occ_imp.params.index,"coef":res_occ_imp.params.values,"p":res_occ_imp.pvalues.values})
occ_imp["coef"]=occ_imp["coef"].round(4)
occ_imp["p"]=occ_imp["p"].round(4)

# Load task-truth Table2 (task-truth)
task_tbl = pd.read_csv("./in_task_vs_occ/TASKTRUTH_reg_importance_task_segment.csv").copy()
# Ensure rounding
task_tbl["coef"]=task_tbl["coef"].round(4)
task_tbl["p"]=task_tbl["p"].round(4)

terms = ["importance","importance:C(task)[T.summary]","importance:C(task)[T.presentation]"]

task_sub = task_tbl[task_tbl["term"].isin(terms)].copy().rename(columns={"coef":"coef_tasktruth","p":"p_tasktruth"})
occ_sub = occ_imp[occ_imp["term"].isin(terms)].copy().rename(columns={"coef":"coef_occtruth_bm","p":"p_occtruth_bm"})

a3 = (pd.DataFrame({"term": terms})
      .merge(task_sub[["term","coef_tasktruth","p_tasktruth"]], on="term", how="left")
      .merge(occ_sub[["term","coef_occtruth_bm","p_occtruth_bm"]], on="term", how="left"))

out_path="./out_task_vs_occ/task_vs_occ_importance_comparison.csv"
a3.to_csv(out_path, index=False, encoding="utf-8-sig")
out_path