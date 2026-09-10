import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.unicode_minus"] = False

# Hardcode final task-truth values provided by user (BM×time, clipped)
T_task = {
    "summary": 0.8407778154885289,
    "instruction": 0.5126995521617606,
    "presentation": 0.4983328701102114,
}

proxy = pd.read_csv("./in_OCCBM_proxy_scatter/proxy_model_summary.csv")[["model","proxy_total"]]
model_summary = pd.read_csv("./in_OCCBM_proxy_scatter/model_performance_error_summary.csv")

model_summary["model"] = model_summary["model"].str.lower()
model_summary = model_summary.merge(proxy, on="model", how="left")

x = "proxy_total"
y = "time_abs_ae_bm"
title = "proxy_otal x MAE(Time-Weighted x BM)"
fname = "proxy_total_vs_time_BM_mae.png"

fig = plt.figure(figsize=(7.2, 5.0))
plt.scatter(model_summary[x], model_summary[y])
for _, r in model_summary.iterrows():
    if pd.isna(r[x]) or pd.isna(r[y]):
        continue
    plt.text(r[x], r[y], str(r["model"]), fontsize=8, ha="left", va="bottom")
plt.title(title)
plt.xlabel(x)
plt.ylabel(y)
plt.grid(True, alpha=0.3)
out = f"./out_OCCBM_proxy_scatter/{fname}"
plt.tight_layout()
plt.savefig(out, dpi=200)
plt.show()
plt.close(fig)