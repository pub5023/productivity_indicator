import pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

model = pd.read_csv("./in_proxy_regression0/proxy_model_summary.csv")
long = pd.read_csv("./in_proxy_regression0/proxy_by_file_long.csv")

# segment assignment (note: case in keys)
def assign_segment(m: str) -> str:
    s=str(m).lower()
    if s in ["gpt52p","gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i","gpt52t","opus","sonnet","haiku"]):
        return "A"
    return "B"

model["segment"] = model["model"].apply(assign_segment)

# overall correlations
sub = model.dropna(subset=["proxy_total","mean_signed_error_bm"]).copy()
pear_r, pear_p = pearsonr(sub["proxy_total"], sub["mean_signed_error_bm"])
spear_r, spear_p = spearmanr(sub["proxy_total"], sub["mean_signed_error_bm"])

# component correlations
rows=[]
for col in ["residual_per_1k","scope_per_1k","caution_per_1k","proxy_total"]:
    r,p = pearsonr(sub[col], sub["mean_signed_error_bm"])
    rs,ps = spearmanr(sub[col], sub["mean_signed_error_bm"])
    rows.append({"x":col,"n":len(sub),"pearson_r":r,"pearson_p":p,"spearman_r":rs,"spearman_p":ps})
corr = pd.DataFrame(rows)
corr_round=corr.copy()
for c in ["pearson_r","pearson_p","spearman_r","spearman_p"]:
    corr_round[c]=corr_round[c].round(4)

corr_round.to_csv("./out_proxy_regression0/proxy_BM.csv", index=False, encoding="utf-8-sig")

# segment-wise correlations
seg_rows=[]
for seg in ["A","B","C"]:
    ssub=model[model["segment"]==seg]
    if len(ssub) >= 3:
        r,p=pearsonr(ssub["proxy_total"], ssub["mean_signed_error_bm"])
        rs,ps=spearmanr(ssub["proxy_total"], ssub["mean_signed_error_bm"])
    else:
        r=p=rs=ps=np.nan
        seg_rows.append({"segment":seg,"n":len(ssub),"pearson_r":r,"pearson_p":p,"spearman_r":rs,"spearman_p":ps})
seg_corr=pd.DataFrame(seg_rows)
for c in ["pearson_r","pearson_p","spearman_r","spearman_p"]:
    seg_corr[c]=seg_corr[c].round(4)

seg_corr.to_csv("./out_proxy_regression0/proxy_total_vs_BM.csv", index=False, encoding="utf-8-sig")

# regression with segment fixed effects (A reference)
df=model.copy()
df["segB"]=(df["segment"]=="B").astype(float)
df["segC"]=(df["segment"]=="C").astype(float)
X=pd.DataFrame({
    "const": 1.0,
    "proxy_total": df["proxy_total"].astype(float),
    "segB": df["segB"],
    "segC": df["segC"],
})
y=df["mean_signed_error_bm"].astype(float)
res=sm.OLS(y,X).fit()

coef=pd.DataFrame({"term":res.params.index,"coef":res.params.values,"p":res.pvalues.values})
coef["coef"]=coef["coef"].round(4); coef["p"]=coef["p"].round(4)

# interaction model
X2=pd.DataFrame({
    "const": 1.0,
    "proxy_total": df["proxy_total"].astype(float),
    "segB": df["segB"],
    "segC": df["segC"],
    "proxy_total:segB": (df["proxy_total"]*df["segB"]).astype(float),
    "proxy_total:segC": (df["proxy_total"]*df["segC"]).astype(float),
})
res2=sm.OLS(y,X2).fit()
coef2=pd.DataFrame({"term":res2.params.index,"coef":res2.params.values,"p":res2.pvalues.values})
coef2["coef"]=coef2["coef"].round(4); coef2["p"]=coef2["p"].round(4)

coef2.sort_values("p").to_csv("./out_proxy_regression0/BM_proxy_total_segment_A.csv", index=False, encoding="utf-8-sig")

# fit summaries
fit_tbl = pd.DataFrame([
    {"model":"overall OLS (no segment)", "R2": float(sm.OLS(y, sm.add_constant(df["proxy_total"])).fit().rsquared)},
    {"model":"+ segment", "R2": float(res.rsquared)},
    {"model":"+ segment + interactions", "R2": float(res2.rsquared)},
])
fit_tbl["R2"]=fit_tbl["R2"].round(4)

fit_tbl.to_csv("./out_proxy_regression0/proxy_R2.csv", index=False, encoding="utf-8-sig")

# show key numbers
{
 "n_models": len(sub),
 "segments": model["segment"].value_counts().to_dict(),
 "overall_pearson_r": float(pear_r),
 "overall_pearson_p": float(pear_p),
 "overall_spearman_r": float(spear_r),
 "overall_spearman_p": float(spear_p),
 "reg_proxy_coef": float(res.params["proxy_total"]),
 "reg_proxy_p": float(res.pvalues["proxy_total"]),
 "reg_segB_coef": float(res.params["segB"]),
 "reg_segB_p": float(res.pvalues["segB"]),
 "reg_segC_coef": float(res.params["segC"]),
 "reg_segC_p": float(res.pvalues["segC"]),
}

model.assign(segment=model["segment"])[[
    "model","segment","proxy_total","mean_signed_error_bm","residual_per_1k","scope_per_1k","caution_per_1k","n_files","chars_total"
    ]].sort_values("proxy_total", ascending=False).to_csv("./out_proxy_regression0/proxy_total_BM_segment_.csv", index=False, encoding="utf-8-sig")

