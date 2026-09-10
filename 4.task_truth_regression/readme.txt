# task_truth_regression_fullcode.py

## Input files:
../input/Time_Weighted_Average_Data_W.csv
../input/Time_Weighted_Average_Data_L.csv
../input/Importance_Weighted_Average_Data_W.csv
../input/Importance_Weighted_Average_Data_L.csv

./in_task_truth_regression_fullcode/proxy_model_summary.csv
./in_task_truth_regression_fullcode/proxy_by_file_long.csv

These are the same as output files from "proxy_new.py".

## Output:
TASKTRUTH_reg_segment_task.csv
TASKTRUTH_reg_importance_task_segment.csv
TASKTRUTH_reg_proxy_task_segment.csv
TASKTRUTH_model_level_errors.csv
TASKTRUTH_task_truth_BM_time_clipped.csv
proxy_deficit.csv

## Note
TASKTRUTH_task_truth_BM_time_clipped.csv
TASKTRUTH_model_level_errors.csv
should be used as an input data as shown below.

---

# figures.py

## Input files:
../input/Time_Weighted_Average_Data_L.csv
../input/Time_Weighted_Average_Data_W.csv
../input/Importance_Weighted_Average_Data_W.csv
../input/Importance_Weighted_Average_Data_L.csv

./in_figures/proxy_model_summary.csv

This is the same as output file from "proxy_new.py".

./in_figures/TASKTRUTH_model_level_errors.csv

This is the same as output file from "task_truth_regression_fullcode.py".

## Output:
TASKTRUTH_SCATTER_importance_vs_task_error.png
TASKTRUTH_SCATTER_proxy_total_vs_model_task_mae.png

---

# proxy_regression2.py

## Input files:
../input/Importance_Weighted_Average_Data_W.csv
./in_proxy_regression/TASKTRUTH_task_truth_BM_time_clipped.csv

This is the same as output file from "task_truth_regression_fullcode.py".

./in_proxy_regression/proxy_model_summary.csv
./in_proxy_regression2/proxy_model_summary_cc.csv
./in_proxy_regression2/proxy_model_summary_segmean.csv
./in_proxy_regression2/proxy_by_file_long.csv

These are the same as output files from "proxy_new.py".

## Output:
TASKTRUTH_proxy_reg_complete_case.csv
TASKTRUTH_proxy_reg_allmodels_proxy0.csv
TASKTRUTH_proxy_reg_allmodels_segmean.csv
proxy_sensitivity_comparison.csv

---

# model_errors

## Input files:
../input/Importance_Weighted_Average_Data_W.csv

./in_model_errors/proxy_model_summary.csv

This is the same as output file from "proxy_new.py".

## Output:
TASKTRUTH_model_level_errors.csv