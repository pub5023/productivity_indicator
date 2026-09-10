import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, linregress
import statsmodels.formula.api as smf

# Load
perf  = pd.read_csv("../0.input_data/LLM_Evaluation_Score20260304.csv")
timeW = pd.read_csv("../0.input_data/Time_Weighted_Average_Data_W.csv")
timeL = pd.read_csv("../0.input_data/Time_Weighted_Average_Data_L.csv")
impW  = pd.read_csv("../0.input_data/Importance_Weighted_Average_Data_W.csv")
impL  = pd.read_csv("../0.input_data/Importance_Weighted_Average_Data_L.csv")


# model sets
models_perf = set(perf["model"].astype(str))
models_time = set(timeW["model"].astype(str))
models_imp  = set(impW["model"].astype(str))
common_models = sorted(list(models_perf & models_time & models_imp))

# Aggregation helper
def summarize_errors(df, ae_cols, re_cols, prefix):
    d = df.copy()
    for c in ae_cols + re_cols:
        d[f"abs_{c}"] = d[c].abs()
    agg_cols = [f"abs_{c}" for c in ae_cols + re_cols]
    agg = d.groupby("model")[agg_cols].mean().reset_index()
    agg = agg.rename(columns={c: f"{prefix}_{c}" for c in agg_cols})
    return agg

time_agg = summarize_errors(timeW, ["ae_bm", "ae_ref4"], ["re_bm", "re_ref4"], "time")
imp_agg  = summarize_errors(impW,  ["ae2_bm", "ae2_ref4"], ["re2_bm", "re2_ref4"], "imp")

merged = perf.merge(time_agg, on="model", how="inner").merge(imp_agg, on="model", how="inner")

# numeric conversion if present
for col in ["Arena Elo", "AAII", "MMLU-Pro", "ARC-AGI"]:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

# Some perf files may not have "Model" display name column; adapt
base_cols = ["model"]
if "Model" in merged.columns:
    base_cols.append("Model")

perf_cols = [c for c in ["Arena Elo","AAII","MMLU-Pro","ARC-AGI"] if c in merged.columns]
err_cols = [
    "time_abs_ae_bm","time_abs_re_bm","time_abs_ae_ref4","time_abs_re_ref4",
    "imp_abs_ae2_bm","imp_abs_re2_bm","imp_abs_ae2_ref4","imp_abs_re2_ref4"
]
summary_cols = base_cols + perf_cols + err_cols

model_summary = merged[summary_cols].sort_values("Arena Elo", ascending=False).reset_index(drop=True)

# Correlations (Elo vs error metrics)
err_only_cols = [c for c in err_cols if c in model_summary.columns]

def corr_table(df, x, ys):
    rows=[]
    for y in ys:
        sub=df[[x,y]].dropna()
        if len(sub) < 3:
            continue
        r_p, p_p = pearsonr(sub[x], sub[y])
        r_s, p_s = spearmanr(sub[x], sub[y])
        lr = linregress(sub[x], sub[y])
        rows.append({
            "y": y, "n": len(sub),
            "pearson_r": float(r_p), "pearson_p": float(p_p),
            "spearman_r": float(r_s), "spearman_p": float(p_s),
            "slope(y~Elo)": float(lr.slope), "slope_p": float(lr.pvalue)
        })
    return pd.DataFrame(rows).sort_values("pearson_r")

corr_elo = corr_table(model_summary, "Arena Elo", err_only_cols)

# Plots
def scatter_with_labels(df, x, y, title, fname):
    fig = plt.figure(figsize=(7.2, 5.0))
    plt.scatter(df[x], df[y])
    for _, r in df.iterrows():
        if pd.isna(r[x]) or pd.isna(r[y]):
            continue
        plt.text(r[x], r[y], str(r["model"]), fontsize=8, ha="left", va="bottom")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)
    out = f"./out_importance_regression/{fname}"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out

p_time_bm = scatter_with_labels(model_summary, "Arena Elo", "time_abs_ae_bm",
                                "Arena Elo × MAE (Time-Weighted × BM)", "elo_vs_mae_time_bm.png")
p_imp_bm = scatter_with_labels(model_summary, "Arena Elo", "imp_abs_ae2_bm",
                               "Arena Elo × MAE (Importance-Weighted × BM)", "elo_vs_mae_imp_bm.png")
p_time_ref4 = scatter_with_labels(model_summary, "Arena Elo", "time_abs_ae_ref4",
                                  "Arena Elo × MAE (Time-Weighted × ref4)", "elo_vs_mae_time_ref4.png")

# Interaction regression (importance x task) from impW
for t in ["summary", "instruction", "presentation"]:
    impW[f"est_red_{t}"] = 1 - impW[f"est_ai_time_{t}"] / impW[f"est_human_time_{t}"]

task_df = pd.concat([
    pd.DataFrame({
        "model": impW["model"], "occupation_id": impW["occupation_id"], "occupation_name": impW["occupation_name"],
        "task": "summary", "importance": impW["task_weight_summary"].astype(float),
        "est_reduction": impW["est_red_summary"].astype(float),
        "true_bm": impW["true_reduction_bm"].astype(float),
        "true_ref4": impW["true_reduction_ref4"].astype(float),
    }),
    pd.DataFrame({
        "model": impW["model"], "occupation_id": impW["occupation_id"], "occupation_name": impW["occupation_name"],
        "task": "instruction", "importance": impW["task_weight_instruction"].astype(float),
        "est_reduction": impW["est_red_instruction"].astype(float),
        "true_bm": impW["true_reduction_bm"].astype(float),
        "true_ref4": impW["true_reduction_ref4"].astype(float),
    }),
    pd.DataFrame({
        "model": impW["model"], "occupation_id": impW["occupation_id"], "occupation_name": impW["occupation_name"],
        "task": "presentation", "importance": impW["task_weight_presentation"].astype(float),
        "est_reduction": impW["est_red_presentation"].astype(float),
        "true_bm": impW["true_reduction_bm"].astype(float),
        "true_ref4": impW["true_reduction_ref4"].astype(float),
    }),
], ignore_index=True)

task_df["signed_error_bm"] = task_df["est_reduction"] - task_df["true_bm"]
task_df["signed_error_ref4"] = task_df["est_reduction"] - task_df["true_ref4"]

task_ref = "summary"
res_bm = smf.ols(f'signed_error_bm ~ importance * C(task, Treatment(reference="{task_ref}"))', data=task_df).fit(cov_type="HC3")
res_rf = smf.ols(f'signed_error_ref4 ~ importance * C(task, Treatment(reference="{task_ref}"))', data=task_df).fit(cov_type="HC3")

def coef_tbl(res, baseline_label):
    cf = pd.DataFrame({
        "baseline": baseline_label,
        "term": res.params.index,
        "coef": res.params.values,
        "se(HC3)": res.bse.values,
        "t": res.tvalues.values,
        "p": res.pvalues.values,
    })
    keep = [
        "Intercept", "importance",
        f'C(task, Treatment(reference="{task_ref}"))[T.instruction]',
        f'C(task, Treatment(reference="{task_ref}"))[T.presentation]',
        f'importance:C(task, Treatment(reference="{task_ref}"))[T.instruction]',
        f'importance:C(task, Treatment(reference="{task_ref}"))[T.presentation]',
    ]
    cf = cf[cf["term"].isin(keep)].copy()
    for col, nd in [("coef",4),("se(HC3)",4),("t",3),("p",4)]:
        cf[col] = cf[col].round(nd)
    return cf

coef_df = pd.concat([coef_tbl(res_bm, "BM"), coef_tbl(res_rf, "ref4")], ignore_index=True)

def slopes_from_res(res, baseline_label):
    p = res.params
    base = float(p["importance"])
    inst = float(p.get(f'importance:C(task, Treatment(reference="{task_ref}"))[T.instruction]', 0.0))
    pres = float(p.get(f'importance:C(task, Treatment(reference="{task_ref}"))[T.presentation]', 0.0))
    out = pd.DataFrame([
        {"baseline": baseline_label, "task":"summary", "slope": base},
        {"baseline": baseline_label, "task":"instruction", "slope": base + inst},
        {"baseline": baseline_label, "task":"presentation", "slope": base + pres},
    ])
    out["slope"] = out["slope"].round(4)
    return out

slopes_df = pd.concat([slopes_from_res(res_bm, "BM"), slopes_from_res(res_rf, "ref4")], ignore_index=True)

# Save updated artifacts
out_model_summary = "./out_importance_regression/model_performance_error_summary.csv"
out_corr = "./out_importance_regression/corr_ArenaElo_vs_errors.csv"
out_coef = "./out_importance_regression/regression_coef_importance_task_BM_ref4.csv"
out_slopes = "./out_importance_regression/regression_slopes_by_task_BM_ref4.csv"

model_summary.to_csv(out_model_summary, index=False)
corr_elo.to_csv(out_corr, index=False)
coef_df.to_csv(out_coef, index=False)
slopes_df.to_csv(out_slopes, index=False)

out_paths = {
    "model_summary": out_model_summary,
    "corr_elo": out_corr,
    "interaction_coef": out_coef,
    "interaction_slopes": out_slopes,
    "plot_time_bm": p_time_bm,
    "plot_imp_bm": p_imp_bm,
    "plot_time_ref4": p_time_ref4,
}
out_paths
