#!/usr/bin/env python3
"""
Reproduce Appendix Table F.1 from the uploaded English wide CSV.

IMPORTANT:
- For the task-truth regression used in Table 2 / Table F.1, this script
  recomputes the unweighted task-level reduction rate as:

      est_reduction_task = 1 - est_ai_time_task / est_human_time_task

Regression:
    task_error ~ C(segment) + C(task)
Reference categories:
    segment = A
    task = instruction
Task-truth values (BM x time, clipped):
    summary      = 0.8408
    instruction  = 0.5127
    presentation = 0.4983

Outputs:
- table_F1_cluster_robustness_from_importanceW_correct.csv
- table_F1_diagnostic_from_importanceW_correct.csv
"""

import argparse
import csv
import math
from collections import OrderedDict

import numpy as np
from scipy import stats

import statsmodels.formula.api as smf
import pandas as pd

SEGMENTS = {
    "gpt51T": "C",
    "gpt52P": "C",
    "gpt52I": "A",
    "gpt52T": "A",
    "sonnet46": "A",
    "sonnet46T": "A",
    "opus46": "A",
    "opus46T": "A",
    "haiku45": "A",
    "gpt35t": "B",
    "gpt4om": "B",
    "gem2:9b": "B",
    "llama32:3b": "B",
    "mist:7b": "B",
    "llama2:13b": "B",
}

TASK_TRUTH = OrderedDict([
    ("summary", 0.8407778154885285),
    ("instruction", 0.512699552161761),
    ("presentation", 0.4983328701102113),
])

TERMS = ["intercept", "S.B", "S.C", "T.pres", "T.sum"]


def fnum(x):
    if x is None or x == "":
        return math.nan
    return float(str(x).replace(",", ""))


def load_task_rows(input_csv):
    rows = []
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            model = rec["model"]
            if model not in SEGMENTS:
                raise ValueError(f"Unknown model id: {model}")
            for task, truth in TASK_TRUTH.items():
                human = fnum(rec[f"est_human_time_{task}"])
                ai = fnum(rec[f"est_ai_time_{task}"])
                if not np.isfinite(human) or human <= 0:
                    raise ValueError(f"Invalid human time: model={model}, occupation={rec['occupation_id']}, task={task}")
                if not np.isfinite(ai):
                    raise ValueError(f"Invalid AI time: model={model}, occupation={rec['occupation_id']}, task={task}")
                est_reduction = 1.0 - ai / human
                rows.append({
                    "model": model,
                    "occupation_id": rec["occupation_id"],
                    "occupation_name": rec.get("occupation_name", ""),
                    "segment": SEGMENTS[model],
                    "task": task,
                    "est_reduction": est_reduction,
                    "task_truth": truth,
                    "task_error": est_reduction - truth,
                })
    return rows


def design_matrix(rows):
    X = []
    y = []
    model_groups = []
    occupation_groups = []
    for r in rows:
        X.append([
            1.0,
            1.0 if r["segment"] == "B" else 0.0,
            1.0 if r["segment"] == "C" else 0.0,
            1.0 if r["task"] == "presentation" else 0.0,
            1.0 if r["task"] == "summary" else 0.0,
        ])
        y.append(r["task_error"])
        model_groups.append(r["model"])
        occupation_groups.append(r["occupation_id"])
    return np.array(X, dtype=float), np.array(y, dtype=float), model_groups, occupation_groups


def ols_fit(X, y):
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    return beta, resid, xtx_inv


def hc3_cov(X, resid, xtx_inv):
    n, k = X.shape
    h = np.sum((X @ xtx_inv) * X, axis=1)
    scaled = resid / (1.0 - h)
    meat = (X * scaled[:, None]).T @ (X * scaled[:, None])
    return xtx_inv @ meat @ xtx_inv


def cluster_cov(X, resid, xtx_inv, groups):
    n, k = X.shape
    group_values = list(OrderedDict((g, None) for g in groups).keys())
    meat = np.zeros((k, k), dtype=float)
    for g in group_values:
        idx = [i for i, gi in enumerate(groups) if gi == g]
        Xg = X[idx, :]
        eg = resid[idx]
        score = Xg.T @ eg
        meat += np.outer(score, score)
    G = len(group_values)
    correction = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    cov = correction * xtx_inv @ meat @ xtx_inv
    return cov, G


def pvalues(beta, se, df):
    tvals = beta / se
    return 2.0 * stats.t.sf(np.abs(tvals), df=df)


def write_outputs(rows, output_csv, diag_csv):
    X, y, model_groups, occupation_groups = design_matrix(rows)
    n, k = X.shape
    beta, resid, xtx_inv = ols_fit(X, y)

    task_df = pd.DataFrame(rows)
    res_seg = smf.ols("task_error ~ C(segment) + C(task)", data=task_df).fit(cov_type="HC3")
    se_hc3 = res_seg.HC3_se.values
    p_hc3 = res_seg.pvalues.values
    
    cov_model, G_model = cluster_cov(X, resid, xtx_inv, model_groups)
    se_model = np.sqrt(np.diag(cov_model))
    p_model = pvalues(beta, se_model, G_model - 1)

    cov_occ, G_occ = cluster_cov(X, resid, xtx_inv, occupation_groups)
    se_occ = np.sqrt(np.diag(cov_occ))
    p_occ = pvalues(beta, se_occ, G_occ - 1)

    table_rows = []
    for i, term in enumerate(TERMS):
        table_rows.append({
            "term": term,
            "coefficient": beta[i],
            "HC3_SE": se_hc3[i],
            "HC3_p": p_hc3[i],
            "model_cluster_SE": se_model[i],
            "model_cluster_p": p_model[i],
            "occupation_cluster_SE": se_occ[i],
            "occupation_cluster_p": p_occ[i],
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    diag = [
        {"item": "n_observations", "value": n},
        {"item": "n_models", "value": len(set(model_groups))},
        {"item": "n_occupations", "value": len(set(occupation_groups))},
        {"item": "n_tasks", "value": len(TASK_TRUTH)},
        {"item": "regression", "value": "task_error ~ C(segment) + C(task)"},
        {"item": "reference_segment", "value": "A"},
        {"item": "reference_task", "value": "instruction"},
        {"item": "task_truth_summary", "value": TASK_TRUTH["summary"]},
        {"item": "task_truth_instruction", "value": TASK_TRUTH["instruction"]},
        {"item": "task_truth_presentation", "value": TASK_TRUTH["presentation"]},
    ]
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "value"])
        writer.writeheader()
        writer.writerows(diag)

    return table_rows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/mnt/data/Importance_Weighted_Average_Data_W.csv")
    ap.add_argument("--out-csv", default="/mnt/data/table_F1_cluster_robustness_from_importanceW.csv")
    ap.add_argument("--out-diag", default="/mnt/data/table_F1_diagnostic_from_importanceW.csv")
    args = ap.parse_args()

    rows = load_task_rows(args.input)
    table_rows, diag = write_outputs(rows, args.out_csv, args.out_diag)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_diag}")
    for row in table_rows:
        print(row)


if __name__ == "__main__":
    main()
