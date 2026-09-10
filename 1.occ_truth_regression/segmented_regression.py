import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

# Load updated data
impW = pd.read_csv("../0.input_data/Importance_Weighted_Average_Data_W.csv")

# Reconstruct task-level reductions
for t in ["summary","instruction","presentation"]:
    impW[f"est_red_{t}"] = 1 - impW[f"est_ai_time_{t}"] / impW[f"est_human_time_{t}"]

task_df = pd.concat([
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "occupation_name": impW["occupation_name"],
        "task": "summary",
        "importance": impW["task_weight_summary"].astype(float),
        "est_reduction": impW["est_red_summary"].astype(float),
        "true_bm": impW["true_reduction_bm"].astype(float),
        "true_ref4": impW["true_reduction_ref4"].astype(float),
    }),
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "occupation_name": impW["occupation_name"],
        "task": "instruction",
        "importance": impW["task_weight_instruction"].astype(float),
        "est_reduction": impW["est_red_instruction"].astype(float),
        "true_bm": impW["true_reduction_bm"].astype(float),
        "true_ref4": impW["true_reduction_ref4"].astype(float),
    }),
    pd.DataFrame({
        "model": impW["model"],
        "occupation_id": impW["occupation_id"],
        "occupation_name": impW["occupation_name"],
        "task": "presentation",
        "importance": impW["task_weight_presentation"].astype(float),
        "est_reduction": impW["est_red_presentation"].astype(float),
        "true_bm": impW["true_reduction_bm"].astype(float),
        "true_ref4": impW["true_reduction_ref4"].astype(float),
    }),
], ignore_index=True)

task_df["signed_error_bm"] = task_df["est_reduction"] - task_df["true_bm"]
task_df["signed_error_ref4"] = task_df["est_reduction"] - task_df["true_ref4"]

# Segment mapping
def assign_segment(model: str) -> str:
    m = str(model).lower()
    if m in ["gpt52p", "gpt51t"]:
        return "C"
    if ("gpt5" in m) or ("opus" in m) or ("sonnet" in m) or ("haiku" in m):
        return "A"
    return "B"

task_df["segment"] = task_df["model"].apply(assign_segment)

# Regressions
task_ref="summary"
base_bm = f'signed_error_bm ~ importance * C(task, Treatment(reference="{task_ref}")) + C(segment)'
full_bm = f'signed_error_bm ~ importance * C(task, Treatment(reference="{task_ref}")) * C(segment)'
base_rf = f'signed_error_ref4 ~ importance * C(task, Treatment(reference="{task_ref}")) + C(segment)'
full_rf = f'signed_error_ref4 ~ importance * C(task, Treatment(reference="{task_ref}")) * C(segment)'

res_bm_base = smf.ols(base_bm, data=task_df).fit(cov_type="HC3")
res_bm_full = smf.ols(full_bm, data=task_df).fit(cov_type="HC3")
res_rf_base = smf.ols(base_rf, data=task_df).fit(cov_type="HC3")
res_rf_full = smf.ols(full_rf, data=task_df).fit(cov_type="HC3")

def extract_terms(res, baseline_label, spec_label):
    p = res.params; se = res.bse; pv = res.pvalues
    rows=[]
    def add(term):
        if term in p.index:
            rows.append({
                "baseline": baseline_label, "spec": spec_label, "term": term,
                "coef": float(p[term]), "se(HC3)": float(se[term]), "p": float(pv[term])
            })
    add("C(segment)[T.B]"); add("C(segment)[T.C]")
    add("importance")
    add(f'importance:C(task, Treatment(reference="{task_ref}"))[T.instruction]')
    add(f'importance:C(task, Treatment(reference="{task_ref}"))[T.presentation]')
    add("importance:C(segment)[T.B]"); add("importance:C(segment)[T.C]")
    add(f'importance:C(task, Treatment(reference="{task_ref}"))[T.instruction]:C(segment)[T.B]')
    add(f'importance:C(task, Treatment(reference="{task_ref}"))[T.instruction]:C(segment)[T.C]')
    return pd.DataFrame(rows)

terms_tbl = pd.concat([
    extract_terms(res_bm_base, "BM", "base"),
    extract_terms(res_bm_full, "BM", "full"),
    extract_terms(res_rf_base, "ref4", "base"),
    extract_terms(res_rf_full, "ref4", "full"),
], ignore_index=True)

fit_tbl = pd.DataFrame([
    {"baseline":"BM", "spec":"base", "n": int(res_bm_base.nobs), "R2": float(res_bm_base.rsquared), "AIC": float(res_bm_base.aic)},
    {"baseline":"BM", "spec":"full", "n": int(res_bm_full.nobs), "R2": float(res_bm_full.rsquared), "AIC": float(res_bm_full.aic)},
    {"baseline":"ref4", "spec":"base", "n": int(res_rf_base.nobs), "R2": float(res_rf_base.rsquared), "AIC": float(res_rf_base.aic)},
    {"baseline":"ref4", "spec":"full", "n": int(res_rf_full.nobs), "R2": float(res_rf_full.rsquared), "AIC": float(res_rf_full.aic)},
])

# Plots
def plot_by_segment(df, ycol, title, fname):
    fig = plt.figure(figsize=(10.8, 3.6))
    segments = ["A","B","C"]
    tasks = ["summary","instruction","presentation"]
    axes = [plt.subplot(1, 3, i+1) for i in range(3)]
    xmin, xmax = df["importance"].min(), df["importance"].max()
    xs = np.linspace(xmin, xmax, 50)
    for ax, seg in zip(axes, segments):
        subseg = df[df["segment"]==seg]
        ax.set_title(f"segment {seg}")
        ax.axhline(0, linewidth=1)
        for task in tasks:
            s = subseg[subseg["task"]==task]
            ax.scatter(s["importance"], s[ycol], alpha=0.6, label=task)
            if len(s) >= 2:
                b, a = np.polyfit(s["importance"], s[ycol], 1)
                ax.plot(xs, a + b*xs)
        ax.set_xlabel("importance")
        ax.set_ylabel("signed error")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    out = f"./out_segmented_regression/{fname}"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out

plot_bm = plot_by_segment(task_df, "signed_error_bm", "Segment-wise: Importance × Task and Error (BM-Based)", "segmented_importance_task_BM.png")
plot_rf = plot_by_segment(task_df, "signed_error_ref4", "Segment-wise: Importance × Task and Error (ref4-Based)", "segmented_importance_task_ref4.png")

# Save tables
terms_path = "./out_segmented_regression/segmented_regression_key_terms.csv"
fit_path = "./out_segmented_regression/segmented_regression_fit.csv"
terms_tbl.to_csv(terms_path, index=False)
fit_tbl.to_csv(fit_path, index=False)