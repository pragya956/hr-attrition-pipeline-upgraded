"""Metrics suited to an imbalanced target, curves and CV summaries."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, make_scorer,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

CLASS_NAMES = ["Stayed", "Left"]

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}
# Metrics allowed as the *model-selection* criterion (threshold-free or imbalance-aware).
SELECTION_METRICS = ["average_precision", "roc_auc", "f1", "balanced_accuracy"]


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _safe(fn, *args, **kwargs):
    """Metric that is undefined for degenerate input (e.g. one class) returns None, not an error."""
    try:
        return _num(fn(*args, **kwargs))
    except ValueError:
        return None


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    report = classification_report(y_true, pred, labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    majority = float(max(np.mean(y_true == 0), np.mean(y_true == 1)))
    return {
        "threshold": float(threshold),
        "accuracy": _safe(accuracy_score, y_true, pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, pred),
        "precision": _safe(precision_score, y_true, pred, zero_division=0),
        "recall": _safe(recall_score, y_true, pred, zero_division=0),
        "f1": _safe(f1_score, y_true, pred, zero_division=0),
        "roc_auc": _safe(roc_auc_score, y_true, proba),
        "average_precision": _safe(average_precision_score, y_true, proba),
        "majority_baseline_accuracy": majority,
        "confusion_matrix": {"labels": CLASS_NAMES, "matrix": cm.tolist(),
                             "tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "classification_report": {k: ({m: _num(v) for m, v in val.items()} if isinstance(val, dict) else _num(val))
                                  for k, val in report.items()},
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
    }


def _thin(arrays, max_points=150):
    n = len(arrays[0])
    if n <= max_points:
        return [np.asarray(a).tolist() for a in arrays]
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return [np.asarray(a)[idx].tolist() for a in arrays]


def compute_curves(y_true, proba) -> dict:
    out = {}
    try:
        fpr, tpr, _ = roc_curve(y_true, proba)
        out["roc"] = dict(zip(("fpr", "tpr"), _thin([fpr, tpr])))
        prec, rec, _ = precision_recall_curve(y_true, proba)
        out["pr"] = dict(zip(("precision", "recall"), _thin([prec, rec])))
        out["pr"]["baseline"] = float(np.mean(np.asarray(y_true) == 1))
    except ValueError:
        out["roc"] = out["pr"] = None
    return out


def summarize_cv(cv_result: dict) -> dict:
    """Mean / std per metric using nan-aware statistics, plus the number of failed folds."""
    summary = {}
    n_folds = len(cv_result["test_accuracy"])
    failed = int(np.sum(np.isnan(cv_result["test_accuracy"])))
    for metric in SCORING:
        vals = np.asarray(cv_result[f"test_{metric}"], dtype=float)
        ok = vals[~np.isnan(vals)]
        summary[metric] = {
            "folds": [_num(v) for v in vals],
            "mean": _num(ok.mean()) if ok.size else None,
            "std": _num(ok.std(ddof=0)) if ok.size else None,
        }
    summary["fit_time_mean"] = _num(np.nanmean(cv_result["fit_time"]))
    summary["n_folds"] = n_folds
    summary["failed_folds"] = failed
    return summary


def threshold_sweep(y_true, proba, thresholds=None) -> list[dict]:
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = (np.asarray(proba) >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": _safe(precision_score, y_true, pred, zero_division=0),
            "recall": _safe(recall_score, y_true, pred, zero_division=0),
            "f1": _safe(f1_score, y_true, pred, zero_division=0),
            "flagged": int(pred.sum()),
        })
    return rows
