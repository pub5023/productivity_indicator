Time_Weighted_Average_Data_W.csv

model ... AI model name
occupation_id ... Occupation ID
occupation_name ... Occupation name
task_weight_summary ... Time ratio of the summarization task
task_weight_instruction ... Time ratio of the instruction document task
task_weight_presentation ... Time ratio of the presentation task
est_human_time_summary ... Human time for article summarization
est_ai_time_summary ... AI time for article summarization
est_reduction_summary ... Reduction rate for article summarization
est_human_time_instruction ... Human time for creating instruction documents
est_ai_time_instruction ... AI time for creating instruction documents
est_reduction_instruction ... Reduction rate for creating instruction documents
est_human_time_presentation ... Human time for creating presentations
est_ai_time_presentation ... AI time for creating presentations
est_reduction_presentation ... Reduction rate for creating presentations
est_reduction_timeweighted ... Sum of time ratio * reduction rate
true_reduction_bm ... Benchmark reduction rate BM (average of ref1, ref2, and ref4)
ae_bm ... est_reduction_timeweighted - true_reduction_bm
re_bm ... ae_bm / true_reduction_bm
true_reduction_ref4 ... Reduction rate in ref4
ae_ref4 ... est_reduction_timeweighted - true_reduction_ref4
re_ref4 ... ae_ref4 / true_reduction_ref4


Time_Weighted_Average_Data_L.csv

model ... AI model name
occupation_id ... Occupation ID
occupation_name ... Occupation name
time_weight_summary ... Time ratio of the summarization task
time_weight_instruction ... Time ratio of the instruction document task
time_weight_presentation ... Time ratio of the presentation task
est_human_time_summary ... Human time for article summarization
est_ai_time_summary ... AI time for article summarization
est_reduction_summary ... Reduction rate for article summarization
est_human_time_instruction ... Human time for creating instruction documents
est_ai_time_instruction ... AI time for creating instruction documents
est_reduction_instruction ... Reduction rate for creating instruction documents
est_human_time_presentation ... Human time for creating presentations
est_ai_time_presentation ... AI time for creating presentations
est_reduction_presentation ... Reduction rate for creating presentations
est_reduction_timeweighted ... Sum of time ratio * reduction rate
label_source ... Source (BM or ref4)
true_reduction ... Benchmark reduction rate (ref4 reduction rate when label_source is ref4)
ae ... est_reduction_timeweighted - true_reduction
re ... ae / true_reduction


Importance_Weighted_Average_Data_W.csv

model ... AI model name
occupation_id ... Occupation ID
occupation_name ... Occupation name
task_weight_summary ... Importance weight of the summarization task
task_weight_instruction ... Importance weight of the instruction document task
task_weight_presentation ... Importance weight of the presentation task
est_human_time_summary ... Estimated human time for article summarization
est_ai_time_summary ... Estimated AI time for article summarization
est_reduction_summary2 ... Estimated reduction rate for article summarization
est_human_time_instruction ... Estimated human time for creating instruction documents
est_ai_time_instruction ... Estimated AI time for creating instruction documents
est_reduction_instruction2 ... Estimated reduction rate for creating instruction documents
est_human_time_presentation ... Estimated human time for creating presentations
est_ai_time_presentation ... Estimated AI time for creating presentations
est_reduction_presentation2 ... Estimated reduction rate for creating presentations
est_reduction_timeweighted2 ...  Sum of time ratio * reduction rate
true_reduction_bm ... Benchmark reduction rate (average of Ref. 1, Ref. 2, and ref4)
ae2_bm ... est_reduction_timeweighted2 - true_reduction_bm
re2_bm ... ae2_bm / true_reduction_bm
true_reduction_ref4 ... Reduction rate in ref4
ae2_ref4 ... est_reduction_timeweighted2 - true_reduction_ref4
re2_ref4 ... true_reduction_ref4 / ae2_ref4


Importance_Weighted_Average_Data_L.csv

model ... AI model name
occupation_id ... Occupation ID
occupation_name ... Occupation name
task_weight_summary ... Importance weight of the summarization task
task_weight_instruction ... Importance weight of the instruction document task
task_weight_presentation ... Importance weight of the presentation task
est_human_time_summary ... Human time for article summarization
est_ai_time_summary ... AI time for article summarization
est_reduction_summary2 ... Reduction rate for article summarization
est_human_time_instruction ... Human time for creating instruction documents
est_ai_time_instruction ... AI time for creating instruction documents
est_reduction_instruction2 ... Reduction rate for creating instruction documents
est_human_time_presentation ... Human time for creating presentations
est_ai_time_presentation ... AI time for creating presentations
est_reduction_presentation2 ... Reduction rate for creating presentations
est_reduction_timeweighted2 ... Sum of time ratio * reduction rate
label_source ... Source (BM or ref4)
true_reduction ... Benchmark reduction rate (ref4 reduction rate when label_source is ref4)
ae2 ... est_reduction_timeweighted2 - true_reduction
re2 ... ae2 / true_reduction
