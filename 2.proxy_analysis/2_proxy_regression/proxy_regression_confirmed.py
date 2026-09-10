import pandas as pd, hashlib
import numpy as np
import statsmodels.api as sm

path="./in_proxy_regression_confirmed/proxy_model_summary.csv"
raw=open(path,"rb").read()
md5=hashlib.md5(raw).hexdigest()
df=pd.read_csv(path)

def assign_segment(model: str) -> str:
    s=str(model).lower()
    if s in ["gpt52p","gpt51t"]:
        return "C"
    if any(k in s for k in ["gpt52i","gpt52t","opus","sonnet","haiku"]):
        return "A"
    return "B"

df["segment"]=df["model"].apply(assign_segment)
counts=df["segment"].value_counts().to_dict()

for c in ["proxy_total","mean_signed_error_bm"]:
    df[c]=pd.to_numeric(df[c], errors="coerce")

df["segB"]=(df["segment"]=="B").astype(float)
df["segC"]=(df["segment"]=="C").astype(float)

X=pd.DataFrame({"const":1.0,"proxy_total":df["proxy_total"],"segB":df["segB"],"segC":df["segC"]})
y=df["mean_signed_error_bm"]
res=sm.OLS(y,X).fit()

coef=pd.DataFrame({"term":res.params.index,"coef":res.params.values,"se":res.bse.values,"t":res.tvalues.values,"p":res.pvalues.values})
coef_round=coef.copy()
for c in ["coef","se","t","p"]:
    coef_round[c]=coef_round[c].round(6)

meta=pd.DataFrame([{"item":"proxy_model_summary.csv md5","value":md5},
                   {"item":"n_models","value":len(df)},
                   {"item":"segments_count","value":counts}])

# Save output
out="./out_proxy_regression_confirmed/CONFIRMED_BM_proxy_total_segment_A.csv"
coef_round.to_csv(out, index=False)
out