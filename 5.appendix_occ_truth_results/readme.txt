# appendix_job_base.py

## Input files:
../input/Time_Weighted_Average_Data_W.csv
../input/Importance_Weighted_Average_Data_W.csv
./in_appendix_job_base/proxy_model_summary.csv

This is the same as output file from "proxy_new.py".

## Output:
APPX_OCCBM_reg_segment_task.csv
APPX_OCCBM_reg_importance_task_segment.csv
APPX_OCCBM_reg_proxy_task_segment.csv
APPX_OCCBM_model_level_errors.csv

---
# task_vs_occ.py

## Input files:
../input/Importance_Weighted_Average_Data_W.csv
../input/Time_Weighted_Average_Data_W.csv

./in_task_vs_occ/TASKTRUTH_reg_importance_task_segment.csv

This is the same as output file from "task_truth_regression_fullcode.py".

## Output:
task_vs_occ_importance_comparison.csv

---
# OCCBM_proxy_scatter.py

## Input files:
./in_OCCBM_proxy_scatter/proxy_model_summary.csv
./in_OCCBM_proxy_scatter/model_performance_error_summary.csv

These are the same as output files from "proxy_new.py".

## Output:
proxy_total_vs_time_BM_mae.png