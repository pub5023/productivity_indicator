# proxy_preparation/proxy_new.py

## Input files:
../input/Time_Weighted_Average_Data_W.csv
./in_proxy/GPT_Response_*.txt
./proxy_keywords.py


## Output:
proxy_by_file_long.csv
proxy_model_summary.csv
proxy_model_summary_cc.csv
proxy_model_summary_segmean.csv
proxy_vs_bias_plot.png

## Note
proxy_model_summary.csv
should be used as an input data as shown below.

---

# proxy_regression/proxy_regression0.py

## Input files:
./in_proxy_regression0/proxy_model_summary.csv
./in_proxy_regression0/proxy_by_file_long.csv

These are the output files from "proxy.py".


## Output:
BM_proxy_total_segment_A.csv
proxy_BM.csv
proxy_R2.csv
proxy_total_BM_segment_.csv
proxy_total_vs_BM.csv

---

# proxy_regression/proxy_regression_confirmed.py

## Input files:
./in_proxy_regression_confirmed/proxy_model_summary.csv

This is the same as output file from "proxy.py".


## Output:
CONFIRMED_BM_proxy_total_segment_A.csv

---

# proxy_regression/centered/centered_regression.py

## Input files:
./centered/in_centered_regression/proxy_model_summary.csv

This is the same as output file from "proxy.py".


## Output:
BM_3_uncentered_segment_A.csv
centered_correlation_table.csv
centered_regression_components.csv
centered_regression_proxy_total.csv
centered_regression_r2.csv