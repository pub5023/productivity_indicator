# task_truth.py

## Input files:
../input/Time_Weighted_Average_Data_W.csv
../input/Importance_Weighted_Average_Data_W.csv

## Output:
task_truth_decomposition.csv
task_truth_reconstruction_detail.csv
task_truth_reconstruction_metrics.csv

## Note
task_truth_reconstruction_metrics.csv
should be used as an input data as shown below.

---
# Figure2_3.py

## Input files:
../input/Importance_Weighted_Average_Data_W.csv

## Output:
Figure2_task_error_by_segment.png
Figure3_task_error_by_segment_task.png

---
# Figure4_task_truth_fit.py

## Input files:
./in_FigureA1_task_truth_fit/task_truth_reconstruction_metrics.csv

This is the same as output file from "task_truth.py".

## Output:
Figure4_task_truth_decomposition_R2fit.png


