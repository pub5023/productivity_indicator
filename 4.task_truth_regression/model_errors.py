import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.unicode_minus"] = False

# Hardcode final task-truth values provided by user (BM×time, clipped)
T_task = {
    "summary": 0.8407778154885289,
    "instruction": 0.5126995521617606,
    "presentation": 0.4983328701102114,
}

impW = pd.read_csv("../0.input_data/Importance_Weighted_Average_Data_W.csv")
proxy = pd.read_csv("./in_model_errors/proxy_model_summary.csv")[["model","proxy_total"]]

for t in ["summary","instruction","presentation"]:
    impW[f"est_red_{t}"] = 1 - impW[f"est_ai_time_{t}"] / impW[f"est_human_time_{t}"]

task_df = pd.concat([
    pd.DataFrame({"model":impW["model"],"task":"summary","est_reduction":impW["est_red_summary"].astype(float)}),
    pd.DataFrame({"model":impW["model"],"task":"instruction","est_reduction":impW["est_red_instruction"].astype(float)}),
    pd.DataFrame({"model":impW["model"],"task":"presentation","est_reduction":impW["est_red_presentation"].astype(float)}),
], ignore_index=True)

task_df["true_task"] = task_df["task"].map(T_task).astype(float)
task_df["task_error"] = task_df["est_reduction"] - task_df["true_task"]
task_df["abs_task_error"] = task_df["task_error"].abs()

def assign_segment(m: str) -> str:
    s=str(m).lower()
    if s in ["gpt52p","gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i","gpt52t","opus","sonnet","haiku"]):
        return "A"
    return "B"

task_df["segment"] = task_df["model"].apply(assign_segment)
task_df["model"] = task_df["model"].str.lower()
task_df = task_df.merge(proxy, on="model", how="left")

model_mae = (task_df.groupby("model", as_index=False)
             .agg(task_mae=("abs_task_error","mean"),
                  task_bias=("task_error","mean"),
                  proxy_total=("proxy_total","first"),
                  segment=("segment","first")))

# Save underlying data
data_path = "./out_model_errors/TASKTRUTH_model_level_errors.csv"
model_mae.to_csv(data_path, index=False, encoding="utf-8-sig")

(data_path)