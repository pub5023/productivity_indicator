import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.formula.api as smf
import os

# Use the task-truth values the user provided (BM×time, clipped)
T_task = {"summary": 0.8407778154885289,
          "instruction": 0.5126995521617606,
          "presentation": 0.4983328701102114}

impW = pd.read_csv("../0.input_data/Importance_Weighted_Average_Data_W.csv")

for t in ["summary","instruction","presentation"]:
    impW[f"est_red_{t}"] = 1 - impW[f"est_ai_time_{t}"] / impW[f"est_human_time_{t}"]

task_df = pd.concat([
    pd.DataFrame({"model":impW["model"],"task":"summary","importance":impW["task_weight_summary"].astype(float),
                  "est_reduction":impW["est_red_summary"].astype(float)}),
    pd.DataFrame({"model":impW["model"],"task":"instruction","importance":impW["task_weight_instruction"].astype(float),
                  "est_reduction":impW["est_red_instruction"].astype(float)}),
    pd.DataFrame({"model":impW["model"],"task":"presentation","importance":impW["task_weight_presentation"].astype(float),
                  "est_reduction":impW["est_red_presentation"].astype(float)}),
], ignore_index=True)

task_df["true_task"] = task_df["task"].map(T_task).astype(float)
task_df["task_error"] = task_df["est_reduction"] - task_df["true_task"]

def assign_segment(m: str) -> str:
    s=str(m).lower()
    if s in ["gpt52p","gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i","gpt52t","opus","sonnet","haiku"]):
        return "A"
    return "B"
task_df["segment"] = task_df["model"].apply(assign_segment)

# Figure 1a
fig = plt.figure(figsize=(7.2,5.0))
data=[task_df.loc[task_df["segment"]==seg,"task_error"].values for seg in ["A","B","C"]]
plt.boxplot(data, labels=["A","B","C"])
plt.axhline(0, linewidth=1)
plt.xlabel("Segment")
plt.ylabel("task_error (est - task_truth)")
plt.title("Figure 2. Task-truth-based error by segment (BM×time)")
plt.grid(True, axis="y", alpha=0.3)
f1a="../out_Figure_2_3/Figure2_task_error_by_segment.png"
plt.tight_layout()
plt.savefig(f1a, dpi=200)
plt.close(fig)

# Figure 1b
fig = plt.figure(figsize=(10.8,3.6))
tasks=["summary","instruction","presentation"]
for i,t in enumerate(tasks,1):
    ax=plt.subplot(1,3,i)
    sub=task_df[task_df["task"]==t]
    data=[sub.loc[sub["segment"]==seg,"task_error"].values for seg in ["A","B","C"]]
    ax.boxplot(data, labels=["A","B","C"])
    ax.axhline(0, linewidth=1)
    ax.set_title(t)
    ax.set_xlabel("Segment")
    if i==1:
        ax.set_ylabel("task_error")
    ax.grid(True, axis="y", alpha=0.3)
plt.suptitle("Figure 3. Error by segment and task (BM×time)", y=1.02)
plt.tight_layout()
f1b="./out_Figure_2_3/Figure3_task_error_by_segment_task.png"
plt.savefig(f1b, dpi=200, bbox_inches="tight")
plt.close(fig)
