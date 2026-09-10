# importance_regression.py

Input files:
../input/LLM_Evaluation_Score20260304.csv
../input/Time_Weighted_Average_Data_W.csv
../input/Time_Weighted_Average_Data_L.csv
../input/Importance_Weighted_Average_Data_W.csv
../input/Importance_Weighted_Average_Data_L.csv

Output: 
corr_ArenaElo_vs_errors.csv
elo_vs_mae_imp_bm.png
elo_vs_mae_time_bm.png
elo_vs_mae_time_ref4.png
model_performance_error_summary.csv
regression_coef_importance_task_BM_ref4.csv
regression_slopes_by_task_BM_ref4.csv


# segmented_regression.py

Input files:
../input/Importance_Weighted_Average_Data_W.csv

Output:
segmented_importance_task_BM.png
segmented_importance_task_ref4.png
segmented_regression_fit.csv
segmented_regression_key_terms.csv