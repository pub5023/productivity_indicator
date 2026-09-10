import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from proxy_keywords import KW_RESIDUAL, KW_SCOPE, KW_CAUTION

mpl.rcParams["font.family"] = "Yu Gothic"
mpl.rcParams["axes.unicode_minus"] = False


# =========================================================
# 1) input
# =========================================================
TEXT_FILES = sorted(glob.glob("./in_proxy_new/LLM_Response_*.txt"))

# Used for combining BM errors
timeW = pd.read_csv("../0.input_data/Time_Weighted_Average_Data_W.csv")
CSV_MODELS = set(timeW["model"].astype(str).str.lower())

# =========================================================
# 2) proxy keywords
# =========================================================

def normalize_text(s: str) -> str:
    return s.lower().replace("　", " ")

def count_keywords(text: str, keywords: list[str]) -> int:
    return sum(text.count(k) for k in keywords)

# =========================================================
# 3) IMPORTANT: Split into blocks by "model:" lines to eliminate double counting
#    - Prevents errors even when gpt35t and gpt4om appear close together in the same file
# =========================================================
BLOCK_START_RE = re.compile(r"(?m)^\s*model\s*[:=]\s*([a-z0-9\-\:\._]+)\b")

def extract_blocks_by_model(text: str) -> dict:
    """
    Returns: {model_key(lower): concatenated_blocks_text}
    """
    text_l = normalize_text(text)
    matches = list(BLOCK_START_RE.finditer(text_l))
    if not matches:
        return {}

    out = {}
    for i, m in enumerate(matches):
        model_key = m.group(1).strip().lower()
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text_l)
        block = text_l[start:end]

        # Use only models present in the CSV (to make naming inconsistencies easier to detect)
        if model_key in CSV_MODELS:
            out.setdefault(model_key, []).append(block)

    return {k: "\n".join(v) for k, v in out.items()}

# =========================================================
# 4) Aggregation: File × Model (long format)
# =========================================================
rows = []
for fp in TEXT_FILES:
    raw = open(fp, "r", encoding="utf-8", errors="ignore").read()
    blocks = extract_blocks_by_model(raw)

    for model_key, ctx in blocks.items():
        denom = max(1, len(ctx))
        rows.append({
            "file": fp,
            "model": model_key,
            "chars": denom,
            "residual_per_1k": count_keywords(ctx, KW_RESIDUAL) / denom * 1000,
            "scope_per_1k":    count_keywords(ctx, KW_SCOPE)    / denom * 1000,
            "caution_per_1k":  count_keywords(ctx, KW_CAUTION)  / denom * 1000,
        })

long_df = pd.DataFrame(rows)
long_df.to_csv("./out_proxy_new/proxy_by_file_long.csv", index=False, encoding="utf-8-sig")

if long_df.empty:
    raise RuntimeError(
        "No models were detected. "
        "Please add `model: <key matching the 'model' column in Time_Weighted_Average_Data_W.csv>` at the beginning of each block."
    )

# =========================================================
# 5) Aggregation: By Model (weighted average)
# =========================================================
model_proxy = (long_df
    .assign(w=long_df["chars"])
    .groupby("model")
    .apply(lambda g: pd.Series({
        "n_files": g["file"].nunique(),
        "chars_total": g["chars"].sum(),
        "residual_per_1k": np.average(g["residual_per_1k"], weights=g["w"]),
        "scope_per_1k":    np.average(g["scope_per_1k"],    weights=g["w"]),
        "caution_per_1k":  np.average(g["caution_per_1k"],  weights=g["w"]),
    }))
    .reset_index()
)

model_proxy["proxy_total"] = model_proxy[["residual_per_1k","scope_per_1k","caution_per_1k"]].mean(axis=1)

# =========================================================
# 6) Merge with BM error (mean_signed_error_bm)
# =========================================================
model_bias = (timeW.groupby(timeW["model"].astype(str).str.lower())["ae_bm"]
              .mean()
              .reset_index()
              .rename(columns={"model":"model","ae_bm":"mean_signed_error_bm"}))

merged = model_proxy.merge(model_bias, on="model", how="left")
merged.to_csv("./out_proxy_new/proxy_model_summary.csv", index=False, encoding="utf-8-sig")

print("\n=== proxy_model_summary（top 10） ===")
print(merged.sort_values("proxy_total", ascending=False).head(10).to_string(index=False))

# =========================================================
# 7) Plot: proxy_total vs BM signed error
# =========================================================
plot_df = merged.dropna(subset=["mean_signed_error_bm"]).copy()

plt.figure(figsize=(7.2, 5.2))
plt.scatter(plot_df["proxy_total"], plot_df["mean_signed_error_bm"])
for _, r in plot_df.iterrows():
    plt.text(r["proxy_total"], r["mean_signed_error_bm"], r["model"], fontsize=9)

plt.axhline(0, linewidth=1)
plt.xlabel("proxy_total (higher = more remaining work / scope / cautious stance)")
plt.ylabel("mean_signed_error_bm (higher = overestimation)")
plt.title("proxy_total and Conservativeness (BM Signed Error)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./out_proxy_new/proxy_vs_bias_plot.png", dpi=200)
plt.show()

print("\nSaved: proxy_by_file_long.csv, proxy_model_summary.csv, proxy_vs_bias_plot.png")

# =========================================================
# 8) Exclude occupations where residual_per_1k, scope_per_1k, and caution_per_1k are 
# all zero (missing) for complete case analysis.
# =========================================================
long_df_copy = long_df.copy()
cols = ["residual_per_1k","scope_per_1k","caution_per_1k"]
mask = (long_df_copy[cols] == 0).all(axis=1)
long_df_copy = long_df_copy.loc[~mask]

model_proxy_2 = (long_df_copy
    .assign(w=long_df_copy["chars"])
    .groupby("model")
    .apply(lambda g: pd.Series({
        "n_files": g["file"].nunique(),
        "chars_total": g["chars"].sum(),
        "residual_per_1k": np.average(g["residual_per_1k"], weights=g["w"]),
        "scope_per_1k":    np.average(g["scope_per_1k"],    weights=g["w"]),
        "caution_per_1k":  np.average(g["caution_per_1k"],  weights=g["w"]),
    }))
    .reset_index()
)

model_proxy_2["proxy_total"] = model_proxy_2[["residual_per_1k","scope_per_1k","caution_per_1k"]].mean(axis=1)

model_proxy_2.to_csv("./out_proxy_new/proxy_model_summary_cc.csv", index=False, encoding="utf-8-sig")

# =========================================================
# 9) Impute missing proxy values using segment-level averages.
# =========================================================

# --- Segment assignment（A/B/C） ---
def assign_segment(m: str) -> str:
    s = str(m).lower()
    if s in ["gpt52p", "gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i", "gpt52t", "opus", "sonnet", "haiku"]):
        return "A"
    return "B"


long_df_copy = long_df.copy()
long_df_copy["segment"] = long_df_copy["model"].apply(assign_segment)
cols = ["residual_per_1k","scope_per_1k","caution_per_1k"]
segment_mean = (
    long_df_copy[cols]
    .replace(0, np.nan)
    .groupby(long_df_copy['segment'])
    .transform('mean')
)

# Replace only the values that are originally 0 with the segment average.
long_df_copy[cols] = long_df_copy[cols].mask(long_df_copy[cols].eq(0), segment_mean)

model_proxy_3 = (long_df_copy
    .assign(w=long_df_copy["chars"])
    .groupby("model")
    .apply(lambda g: pd.Series({
        "n_files": g["file"].nunique(),
        "chars_total": g["chars"].sum(),
        "residual_per_1k": np.average(g["residual_per_1k"], weights=g["w"]),
        "scope_per_1k":    np.average(g["scope_per_1k"],    weights=g["w"]),
        "caution_per_1k":  np.average(g["caution_per_1k"],  weights=g["w"]),
    }))
    .reset_index()
)

model_proxy_3["proxy_total"] = model_proxy_3[["residual_per_1k","scope_per_1k","caution_per_1k"]].mean(axis=1)
model_proxy_3.to_csv("./out_proxy_new/proxy_model_summary_segmean.csv", index=False, encoding="utf-8-sig")