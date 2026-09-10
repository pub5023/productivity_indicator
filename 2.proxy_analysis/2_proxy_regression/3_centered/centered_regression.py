import pandas as pd, numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv("./in_centered_regression/proxy_model_summary.csv")

def assign_segment(m: str) -> str:
    s=str(m).lower()
    if s in ["gpt52p","gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i","gpt52t","opus","sonnet","haiku"]):
        return "A"
    return "B"

df["segment"] = df["model"].apply(assign_segment)

# Center within segment
for col in ["proxy_total","residual_per_1k","scope_per_1k","caution_per_1k"]:
    df[f"{col}_centered"] = df[col] - df.groupby("segment")[col].transform("mean")

# segment dummies (A reference)
df["segB"] = (df["segment"]=="B").astype(float)
df["segC"] = (df["segment"]=="C").astype(float)

y = df["mean_signed_error_bm"].astype(float)

def run_ols(X_df):
    X = sm.add_constant(X_df, has_constant="add")
    res = sm.OLS(y, X).fit()
    out = pd.DataFrame({"term": res.params.index, "coef": res.params.values, "p": res.pvalues.values})
    out["coef"] = out["coef"].round(4)
    out["p"] = out["p"].round(4)
    return res, out

# Model 1: centered proxy_total + segment
res1, coef1 = run_ols(df[["proxy_total_centered","segB","segC"]])

# Model 2: centered components + segment
res2, coef2 = run_ols(df[["residual_per_1k_centered","scope_per_1k_centered","caution_per_1k_centered","segB","segC"]])

# Model 3: uncentered components + segment (comparison)
res3, coef3 = run_ols(df[["residual_per_1k","scope_per_1k","caution_per_1k","segB","segC"]])

# R2 comparison
fit = pd.DataFrame([
    {"spec":"proxy_total only (no segment)", "R2": sm.OLS(y, sm.add_constant(df["proxy_total"])).fit().rsquared},
    {"spec":"proxy_total + segment", "R2": sm.OLS(y, sm.add_constant(df[["proxy_total","segB","segC"]])).fit().rsquared},
    {"spec":"proxy_total_centered + segment", "R2": res1.rsquared},
    {"spec":"3 components centered + segment", "R2": res2.rsquared},
    {"spec":"3 components uncentered + segment", "R2": res3.rsquared},
])
fit["R2"] = fit["R2"].round(4)

# Correlations for highlighted residual_per_1k and centered variants
def corr_one(xcol, label):
    sub = df[[xcol,"mean_signed_error_bm"]].dropna()
    r,p = pearsonr(sub[xcol], sub["mean_signed_error_bm"])
    rs,ps = spearmanr(sub[xcol], sub["mean_signed_error_bm"])
    return {"x":label,"n":len(sub),"pearson_r":r,"pearson_p":p,"spearman_r":rs,"spearman_p":ps}

corr_tbl = pd.DataFrame([
    corr_one("residual_per_1k","residual_per_1k"),
    corr_one("residual_per_1k_centered","residual_per_1k_centered"),
    corr_one("proxy_total","proxy_total"),
    corr_one("proxy_total_centered","proxy_total_centered"),
])
for c in ["pearson_r","pearson_p","spearman_r","spearman_p"]:
    corr_tbl[c] = corr_tbl[c].round(4)

# Save outputs
coef1_path = "./out_centered_regression/centered_regression_proxy_total.csv"
coef2_path = "./out_centered_regression/centered_regression_components.csv"
coef3_path = "./out_centered_regression/BM_3_uncentered_segment_A.csv"
fit_path = "./out_centered_regression/centered_regression_r2.csv"
corr_path = "./out_centered_regression/centered_correlation_table.csv"

coef1.to_csv(coef1_path, index=False)
coef2.to_csv(coef2_path, index=False)
coef3.to_csv(coef3_path, index=False)
fit.to_csv(fit_path, index=False)
corr_tbl.to_csv(corr_path, index=False)

{"coef1":coef1_path,"coef2":coef2_path,"fit":fit_path,"corr":corr_path,
 "proxy_centered_coef": float(res1.params["proxy_total_centered"]),
 "proxy_centered_p": float(res1.pvalues["proxy_total_centered"]),
 "residual_centered_coef": float(res2.params.get("residual_per_1k_centered", np.nan)),
 "residual_centered_p": float(res2.pvalues.get("residual_per_1k_centered", np.nan)),
 "R2_proxy_centered_seg": float(res1.rsquared),
 "R2_components_centered_seg": float(res2.rsquared)}