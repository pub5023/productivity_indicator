Time_Weighted_Average_Data_W.csv

model ... AIのモデル名
occupation_id ... 職業の番号
occupation_name ... 職業名
task_weight_summary ... 要約タスクの時間比率
task_weight_instruction ... 指示書タスクの時間比率
task_weight_presentation ... プレゼンタスクの時間比率
est_human_time_summary ... 記事要約 人間の時間
est_ai_time_summary ... 記事要約 AIの時間
est_reduction_summary ... 記事要約 削減率
est_human_time_instruction ... 指示書作成 人間の時間
est_ai_time_instruction ... 指示書作成 AIの時間
est_reduction_instruction ... 指示書作成 削減率
est_human_time_presentation ... プレゼン作成 人間の時間
est_ai_time_presentation ... プレゼン作成 AIの時間
est_reduction_presentation ... プレゼン作成 削減率
est_reduction_timeweighted ... 時間比率*削減率の合計
true_reduction_bm ... ベンチマーク削減率（ref.1, ref2, ref.4の平均）
ae_bm ... est_reduction_timeweighted - true_reduction_bm
re_bm ... ae_bm / true_reduction_bm
true_reduction_ref4 ... ref4の削減率
ae_ref4 ... est_reduction_timeweighted - true_reduction_ref4
re_ref4 ... ae_ref4 / true_reduction_ref4

Time_Weighted_Average_Data_L.csv

model ... AIのモデル名
occupation_id ... 職業の番号
occupation_name ... 職業名
time_weight_summary ... 要約タスクの時間比率
time_weight_instruction ... 指示書タスクの時間比率
time_weight_presentation ... プレゼンタスクの時間比率
est_human_time_summary ... 記事要約 人間の時間
est_ai_time_summary ... 記事要約 AIの時間
est_reduction_summary ... 記事要約 削減率
est_human_time_instruction ... 指示書作成 人間の時間
est_ai_time_instruction ... 指示書作成 AIの時間
est_reduction_instruction ... 指示書作成 削減率
est_human_time_presentation ... プレゼン作成 人間の時間
est_ai_time_presentation ... プレゼン作成 AIの時間
est_reduction_presentation ... プレゼン作成 削減率
est_reduction_timeweighted ... 時間比率*削減率の合計
label_source ... ソース（BMまたはref4）
true_reduction ... ベンチマーク削減率（label_sourceがref4の時はref4削減率）
ae ... est_reduction_timeweighted  - true_reduction
re ... ae / true_reduction


Importance_Weighted_Average_Data_W.csv

model ... AIのモデル名
occupation_id ... 職業の番号
occupation_name ... 職業名
task_weight_summary ... 要約タスクの重要度
task_weight_instruction ... 指示書タスクの重要度
task_weight_presentation ... プレゼンタスクの重要度
est_human_time_summary ... 記事要約 人間の時間
est_ai_time_summary ... 記事要約 AIの時間
est_reduction_summary2 ... 記事要約 削減率
est_human_time_instruction ... 指示書作成 人間の時間
est_ai_time_instruction ... 指示書作成 AIの時間
est_reduction_instruction2 ... 指示書作成 削減率
est_human_time_presentation ... プレゼン作成 人間の時間
est_ai_time_presentation ... プレゼン作成 AIの時間
est_reduction_presentation2 ... プレゼン作成 削減率
est_reduction_timeweighted2 ... 時間比率*削減率の合計
true_reduction_bm ... ベンチマーク削減率（ref1, ref2, ref4の平均）
ae2_bm ... est_reduction_timeweighted2 - true_reduction_bm
re2_bm ... ae2_bm / true_reduction_bm
true_reduction_ref4 ... ref4の削減率
ae2_ref4 ... est_reduction_timeweighted2 - true_reduction_ref4
re2_ref4 ... true_reduction_ref4 / ae2_ref4


Importance_Weighted_Average_Data_L.csv

model ... AIのモデル名
occupation_id ... 職業の番号
occupation_name ... 職業名
task_weight_summary ... 要約タスクの重要度
task_weight_instruction ... 指示書タスクの重要度
task_weight_presentation ... プレゼンタスクの重要度
est_human_time_summary ... 記事要約 人間の時間
est_ai_time_summary ... 記事要約 AIの時間
est_reduction_summary2 ... 記事要約 削減率
est_human_time_instruction ... 指示書作成 人間の時間
est_ai_time_instruction ... 指示書作成 AIの時間
est_reduction_instruction2 ... 指示書作成 削減率
est_human_time_presentation ... プレゼン作成 人間の時間
est_ai_time_presentation ... プレゼン作成 AIの時間
est_reduction_presentation2 ... プレゼン作成 削減率
est_reduction_timeweighted2 ... 時間比率*削減率の合計
label_source ... ソース（BMまたはref4）
true_reduction ... ベンチマーク削減率（label_sourceがref4の時はref4削減率）
ae2 ... est_reduction_timeweighted2 - true_reduction
re2 ... ae2 / true_reduction