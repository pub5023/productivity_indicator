# Title:
Technical materials on the manuscript entitled:
"Estimation Style Matters in AI Productivity Prediction: Analyzing
LLM Estimation Errors Across Business Occupations"

# Digest:
This repository contains all materials (prompts, programs, and output data) 
in order to clarify the details of datasets, procedures, and outputs, 
explained in the manuscript. 
You can find detailed research logs in the research notebook (pdf file) in Documents folder.

# Structure
The numbers attached to the folder names correspond to the folder and file numbers referenced in the "research notebook" (Not in the "paper"). 
For a detailed list of file numbers, please refer to File_index_eng_version.xlsx.
Folders:
- (1) Documents
- (2) 0.input_data
- (3) 1.occ_truth_regression
- (4) 2.proxy_analysis
- (5) 3.task_truth_decomposition
- (6) 4.task_truth_regression
- (7) 5.appendix_occ_truth_results
- (8) 6.Table_F1

# Descriptions:
- (1) Documents
This folder contains the detailed log‑note PDF, as well as the file index and screenshots from the Chatbot Arena used in the research. 
The original research data are in Japanese, and the log notes are translations of the Japanese originals.

- (2) 0.input_data
This folder contains the reference data for task‑ and occupation‑specific weights, as well as the estimated values output by each LLM. 
It also includes the original and English‑translated versions of the task descriptions for each occupation. 
The Japanese files used in the original computations are stored in the “JPN” subfolder.

- (3) 1.occ_truth_regression
This folder contains the regression analyses based on the occupation‑level ground‑truth values. 
The analyses primarily examine the correlation with Elo scores and the interactions involving importance measures.
folders:
1_out_importance_regression (Elo analysis)
2_out_segmented_regression (importance interaction analysis)

- (4) 2.proxy_analysis
This folder contains the analyses related to the proxy approach, including the construction of proxy variables, proxy regressions (with segment interactions), 
and centering procedures.
folders:
1_proxy_preparation (construction of proxy variables)
2_proxy_regression (segment interactions)
2_proxy_regression\3_centered (centered analysis)

- (5) 3.task_truth_decomposition
This folder relates to the decomposition of task‑level ground‑truth values and the validation of their reconstruction.
folders:
1_out_task_truth
2_out_Figure_2_3
3_out_Figure4_task_truth_fit


- (6) 4.task_truth_regression
This folder contains the regression analyses based on task‑level ground‑truth values, as well as the proxy sensitivity analyses.
folders:
1_out_task_truth_regression_fullcode
2_out_proxy_regression2
3_out_figures
4_out_model_errors


- (7) 5.appendix_occ_truth_results
This folder contains various regression analyses based on occupation‑level ground‑truth values for the appendix.
folders:
1_out_appendix_job_base
2_out_task_vs_occ
3_out_OCCBM_proxy_scatter


- (8) 6.Table_F1
This folder contains the computations for "Appendix F of the paper", including the cluster‑robust standard errors (CRSE).
folder:
out_table_F1_from_importanceW_robustness

[EOF]