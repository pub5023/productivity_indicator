import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os

PATH_TIME_L = "../0.input_data/Time_Weighted_Average_Data_L.csv"
PATH_TIME_W = "../0.input_data/Time_Weighted_Average_Data_W.csv"
PATH_IMP_W  = "../0.input_data/Importance_Weighted_Average_Data_W.csv"
PATH_IMP_L  = "../0.input_data/Importance_Weighted_Average_Data_L.csv"
PATH_PROXY  = "./in_figures/proxy_model_summary.csv"

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

# =========================================================
# 4) Reconstruct task-level estimated reduction rates from importance-weighted W and create task_df
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
    task_df = task_df.merge(proxy, on="model", how="left")
else:
    task_df["proxy_total"] = np.nan


# Reload outputs created by the pipeline if present
task_df_preview = task_df.copy()

# Plot 1: importance vs task_error by task
fig = plt.figure(figsize=(7.2, 5.0))
for t, sub in task_df_preview.groupby("task"):
    plt.scatter(sub["importance"], sub["task_error"], alpha=0.5, label=t)
plt.axhline(0, linewidth=1)
plt.xlabel("importance ")
plt.ylabel("task_error = est - task_truth")
plt.title("Task-truth based: importance vs task_error")
plt.grid(True, alpha=0.3)
plt.legend()
p1 = "./out_figures/TASKTRUTH_SCATTER_importance_vs_task_error.png"
plt.tight_layout()
plt.savefig(p1, dpi=200)
plt.close(fig)

# Plot 2: proxy_total vs model task_mae (model-level)
model_mae2 = pd.read_csv("./in_figures/TASKTRUTH_model_level_errors.csv")
subm = model_mae2.dropna(subset=["proxy_total"]).copy()

fig = plt.figure(figsize=(7.2, 5.0))
plt.scatter(subm["proxy_total"], subm["task_mae"], alpha=0.8)
for _, r in subm.iterrows():
    plt.text(r["proxy_total"], r["task_mae"], r["model"], fontsize=8)
plt.xlabel("proxy_total")
plt.ylabel("task_mae (mean |est-task_truth|)")
plt.title("Task-truth based: proxy_total vs model task_mae")
plt.grid(True, alpha=0.3)
p2 = "./out_figures/TASKTRUTH_SCATTER_proxy_total_vs_model_task_mae.png"
plt.tight_layout()
plt.savefig(p2, dpi=200)
plt.close(fig)
