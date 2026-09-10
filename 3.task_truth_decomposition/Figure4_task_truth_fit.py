import pandas as pd, glob, os, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.unicode_minus"] = False

# Appendix Figure A1 using reconstruction metrics if exists
figA1=None
metrics_path="./in_Figure4_task_truth_fit/task_truth_reconstruction_metrics.csv"
if os.path.exists(metrics_path):
    met=pd.read_csv(metrics_path)
    col="R2_clipped" if "R2_clipped" in met.columns else "R2"
    labels=[f'{r["baseline"]}-{r["weights"]}' for _, r in met.iterrows()]
    vals=met[col].astype(float).values
    fig=plt.figure(figsize=(7.2,4.6))
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.ylabel(col)
    plt.title("Task-truth decomposition fit")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    figA1="/mnt/data/Figure4_task_truth_decomposition_R2fit.png"
    plt.tight_layout()
    plt.savefig(figA1, dpi=200)
    plt.close(fig)

(figA1)