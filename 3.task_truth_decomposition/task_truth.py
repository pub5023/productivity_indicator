import pandas as pd
import numpy as np

# Load fixed datasets
timeW = pd.read_csv("../0.input_data/Time_Weighted_Average_Data_W.csv")
impW  = pd.read_csv("../0.input_data/Importance_Weighted_Average_Data_W.csv")

# ID normalization
timeW["occupation_id"] = timeW["occupation_id"].astype(str).str.strip()
impW["occupation_id"]  = impW["occupation_id"].astype(str).str.strip()

# groupby
occ_time = (
    timeW.groupby("occupation_id", as_index=False)
    .agg({
        "task_weight_summary": "mean",
        "task_weight_instruction": "mean",
        "task_weight_presentation": "mean",
        "true_reduction_bm": "mean",
        "true_reduction_ref4": "mean",
    })
)

# In importance-weighted W, task_weight_* columns represent importance (0-1)
occ_imp = (
    impW.groupby(["occupation_id", "occupation_name"], as_index=False)
    .agg(
        importance_summary=("task_weight_summary", "first"),
        importance_instruction=("task_weight_instruction", "first"),
        importance_presentation=("task_weight_presentation", "first"),
    )
)

occ = occ_time.merge(occ_imp, on=["occupation_id"], how="inner")

# Build weight matrices
W_time = occ[
    ["task_weight_summary", "task_weight_instruction", "task_weight_presentation"]
].to_numpy(dtype=float)

W_imp_raw = W_time * occ[
    ["importance_summary", "importance_instruction", "importance_presentation"]
].to_numpy(dtype=float)

sums = W_imp_raw.sum(axis=1, keepdims=True)
W_imp = np.where(sums > 0, W_imp_raw / sums, W_time)  # fallback

def solve_task_truth(W: np.ndarray, T: np.ndarray):
    t_hat, residuals, rank, s = np.linalg.lstsq(W, T, rcond=None)
    t_hat = t_hat.flatten()
    t_clip = np.clip(t_hat, 0, 1)

    T_hat = (W @ t_hat.reshape(-1, 1)).flatten()
    T_hat_clip = (W @ t_clip.reshape(-1, 1)).flatten()

    def metrics(y, yhat):
        mae = np.mean(np.abs(yhat - y))
        rmse = np.sqrt(np.mean((yhat - y) ** 2))
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return mae, rmse, r2

    mae, rmse, r2 = metrics(T, T_hat)
    mae_c, rmse_c, r2_c = metrics(T, T_hat_clip)

    return {
        "t_hat": t_hat,
        "t_clip": t_clip,
        "T_hat": T_hat,
        "T_hat_clip": T_hat_clip,
        "metrics": (mae, rmse, r2, mae_c, rmse_c, r2_c),
        "rank": rank,
    }

T_bm = occ["true_reduction_bm"].to_numpy(dtype=float)
T_ref4 = occ["true_reduction_ref4"].to_numpy(dtype=float)

sol = {
    ("BM", "time"): solve_task_truth(W_time, T_bm),
    ("BM", "importance"): solve_task_truth(W_imp, T_bm),
    ("ref4", "time"): solve_task_truth(W_time, T_ref4),
    ("ref4", "importance"): solve_task_truth(W_imp, T_ref4),
}

# Table 1: estimated task truths
rows = []
for (baseline, wtype), s in sol.items():
    t = s["t_hat"]
    tc = s["t_clip"]
    rows.append(
        {
            "baseline": baseline,
            "weights": wtype,
            "rank": s["rank"],
            "summary_t_hat": t[0],
            "instruction_t_hat": t[1],
            "presentation_t_hat": t[2],
            "summary_t_clip": tc[0],
            "instruction_t_clip": tc[1],
            "presentation_t_clip": tc[2],
        }
    )

task_truth = pd.DataFrame(rows)
for c in task_truth.columns:
    if c not in ["baseline", "weights", "rank"]:
        task_truth[c] = task_truth[c].round(4)

print("(Fixed version) Task ground truth decomposition: Estimated task ground truth (BM/ref4 × two types of weights)")
task_truth
task_truth.to_csv("./out_task_truth/task_truth_decomposition.csv", index=False, encoding="utf-8-sig")

# Table 2: reconstruction accuracy
mrows = []
for (baseline, wtype), s in sol.items():
    mae, rmse, r2, mae_c, rmse_c, r2_c = s["metrics"]
    mrows.append(
        {
            "baseline": baseline,
            "weights": wtype,
            "rank": s["rank"],
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAE_clipped": mae_c,
            "RMSE_clipped": rmse_c,
            "R2_clipped": r2_c,
        }
    )

metrics_df = pd.DataFrame(mrows)
for c in ["MAE", "RMSE", "R2", "MAE_clipped", "RMSE_clipped", "R2_clipped"]:
    metrics_df[c] = metrics_df[c].round(6)

print("(Fixed version) Reconstruction accuracy of decomposition (reproducing job-level ground truth)")
metrics_df
metrics_df.to_csv("./out_task_truth/task_truth_reconstruction_metrics.csv", index=False, encoding="utf-8-sig")

# Table 3: occupation-level reconstruction detail (using clipped t)
detail_rows = []
for (baseline, wtype), s in sol.items():
    T = T_bm if baseline == "BM" else T_ref4
    That = s["T_hat_clip"]
    for occ_id, occ_name, t_true, t_hat in zip(
        occ["occupation_id"], occ["occupation_name"], T, That
    ):
        detail_rows.append(
            {
                "baseline": baseline,
                "weights": wtype,
                "occupation_id": occ_id,
                "occupation_name": occ_name,
                "true": t_true,
                "reconstructed": t_hat,
                "abs_error": abs(t_hat - t_true),
            }
        )
detail = pd.DataFrame(detail_rows)
for c in ["true", "reconstructed", "abs_error"]:
    detail[c] = detail[c].round(6)

worst = (
    detail.sort_values("abs_error", ascending=False)
    .groupby(["baseline", "weights"])
    .head(5)
    .reset_index(drop=True)
)


print("(Fixed version) Jobs with large reconstruction errors (Top 5: baseline × weights)")
worst
detail.to_csv("./out_task_truth/task_truth_reconstruction_detail.csv", index=False, encoding="utf-8-sig")