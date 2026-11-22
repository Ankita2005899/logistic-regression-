import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)
from .data_preprocessing import prepare_data
from .model_training import load_pipeline
import os

def get_metrics(dataset_key="student-mat.csv", pipeline=None, X_test=None, y_test=None) -> dict:
    if pipeline is None:
        pipeline = load_pipeline(dataset_key)
        if pipeline is None:
            return {}
        _, X_test, _, y_test, _ = prepare_data(dataset_key)

    y_pred = pipeline.predict(X_test)
    classes = list(pipeline.named_steps["model"].classes_)
    pass_i = classes.index("PASS")
    y_proba = pipeline.predict_proba(X_test)[:, pass_i]
    y_bin = y_test.map({"PASS": 1, "FAIL": 0})

    fpr, tpr, _ = roc_curve(y_bin, y_proba)
    cm = confusion_matrix(y_test, y_pred, labels=["PASS", "FAIL"])

    tn_fail = cm[1, 1]   # actual FAIL predicted FAIL
    fp_fail = cm[1, 0]   # actual FAIL predicted PASS
    baseline = float((y_test == "PASS").mean())  # always predict PASS (majority-class style baseline)

    return {
        "specificity": tn_fail / (tn_fail + fp_fail) if (tn_fail + fp_fail) else 0.0,
        "baseline_accuracy": max(baseline, 1 - baseline),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="PASS", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="PASS", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="PASS", zero_division=0),
        "roc_auc": roc_auc_score(y_bin, y_proba),
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
        "y_test": y_test,
        "report": classification_report(y_test, y_pred, target_names=["PASS", "FAIL"], output_dict=True),
    }
