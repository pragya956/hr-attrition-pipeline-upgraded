"""Training pipeline: stratified split -> stratified CV model comparison -> single test evaluation."""
from __future__ import annotations

import hashlib
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (ParameterGrid, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)

from ml import evaluate as ev
from ml.preprocessing import (PARAM_GRIDS, build_model_pipelines, map_feature_names, prepare_xy,
                              select_feature_columns, split_column_types)

log = logging.getLogger(__name__)
ARTIFACT_VERSION = 3
REQUIRED_KEYS = {"version", "pipeline", "model_name", "schema", "cv_results", "test_metrics",
                 "threshold", "trained_at", "seed", "selection_metric", "dataset_signature",
                 "sklearn_version"}


class TrainingError(RuntimeError):
    pass


class ArtifactError(RuntimeError):
    """Missing, corrupt or incompatible model artifact."""


def dataset_signature(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(str(df.shape).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def build_schema(X_train: pd.DataFrame, num_cols, cat_cols) -> dict:
    features = []
    for c in num_cols:
        s = X_train[c]
        integral = bool(np.all(np.mod(s.dropna(), 1) == 0))
        features.append({
            "name": c, "kind": "numeric", "min": float(s.min()), "max": float(s.max()),
            "default": float(s.median()), "integer": integral,
            "ordinal": bool(integral and s.nunique() <= 10),
        })
    for c in cat_cols:
        s = X_train[c]
        features.append({"name": c, "kind": "categorical", "options": sorted(map(str, s.dropna().unique())),
                         "default": str(s.mode().iloc[0])})
    return {"features": features, "numeric": list(num_cols), "categorical": list(cat_cols),
            "columns": list(num_cols) + list(cat_cols)}


def _feature_importance(fitted: dict, best_name, best_pipe, X_test, y_test, num_cols, cat_cols, seed, log_fn):
    result = {}
    # --- Logistic regression coefficients (numeric features are standardised => per 1 SD) ---
    lr = fitted.get("Logistic Regression")
    if lr is not None:
        names = lr.named_steps["preprocess"].get_feature_names_out()
        mapped = map_feature_names(names, num_cols, cat_cols)
        coefs = lr.named_steps["model"].coef_.ravel()
        rows = []
        for m, c in zip(mapped, coefs):
            label = m["original"] if m["level"] is None else f'{m["original"]} = {m["level"]}'
            rows.append({"feature": label, "original": m["original"], "coefficient": float(c),
                         "odds_ratio": float(np.exp(np.clip(c, -20, 20)))})
        rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
        agg = {}
        for r in rows:
            agg[r["original"]] = agg.get(r["original"], 0.0) + abs(r["coefficient"])
        result["logistic_regression"] = {
            "coefficients": rows[:25],
            "by_feature": sorted([{"feature": k, "importance": v} for k, v in agg.items()],
                                 key=lambda r: r["importance"], reverse=True)[:20],
        }
    # --- Random forest impurity importances, summed back to original features ---
    rf = fitted.get("Random Forest")
    if rf is not None:
        names = rf.named_steps["preprocess"].get_feature_names_out()
        mapped = map_feature_names(names, num_cols, cat_cols)
        agg = {}
        for m, imp in zip(mapped, rf.named_steps["model"].feature_importances_):
            agg[m["original"]] = agg.get(m["original"], 0.0) + float(imp)
        result["random_forest"] = {"by_feature": sorted(
            [{"feature": k, "importance": v} for k, v in agg.items()],
            key=lambda r: r["importance"], reverse=True)[:20]}
    # --- Permutation importance on held-out data (model-agnostic, uses raw readable columns) ---
    log_fn(f"Computing permutation importance for {best_name} on the held-out set...")
    pi = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                random_state=seed, n_jobs=None)
    items = [{"feature": f, "mean": float(m), "std": float(s)}
             for f, m, s in zip(X_test.columns, pi.importances_mean, pi.importances_std)]
    items.sort(key=lambda r: r["mean"], reverse=True)
    result["permutation"] = {"model": best_name, "scoring": "roc_auc", "items": items[:20]}
    return result


def train_pipeline(df: pd.DataFrame, n_splits: int = 5, seed: int = 42, class_weight: bool = True,
                   tune: bool = False, selection_metric: str = "average_precision",
                   test_size: float = 0.2, threshold: float = 0.5,
                   log_fn=lambda msg: None, progress_fn=lambda frac: None) -> dict:
    """Train all models, select with CV only, evaluate the selected model once on the test set."""
    if selection_metric not in ev.SELECTION_METRICS:
        raise TrainingError(f"selection_metric must be one of {ev.SELECTION_METRICS}")
    if not 2 <= n_splits <= 10:
        raise TrainingError("n_splits must be between 2 and 10")

    X, y = prepare_xy(df)
    # 1) Stratified hold-out split. The test set is untouched until the final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    if y_train.value_counts().min() < n_splits:
        raise TrainingError("The minority class has fewer samples than CV folds; lower the fold count.")
    cols = select_feature_columns(X_train)
    X_train, X_test = X_train[cols], X_test[cols]
    num_cols, cat_cols = split_column_types(X_train)
    log_fn(f"Stratified split: {len(X_train)} train / {len(X_test)} test rows "
           f"(attrition rate train {y_train.mean():.1%}, test {y_test.mean():.1%}).")
    log_fn(f"Features: {len(num_cols)} numerical, {len(cat_cols)} categorical (one-hot encoded inside the pipeline).")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipelines = build_model_pipelines(num_cols, cat_cols, seed, class_weight)
    total_steps = len(pipelines) + 3
    done = 0
    cv_results, fitted, best_params = {}, {}, {}

    for name, pipe in pipelines.items():
        estimator = pipe
        if tune:
            grid = PARAM_GRIDS[name]
            n_iter = min(6, len(ParameterGrid(grid)))
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            estimator = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring=selection_metric, cv=inner,
                                           random_state=seed, refit=True, error_score=np.nan)
            log_fn(f"{name}: nested CV ({n_splits} outer folds, 3 inner folds, {n_iter} candidates).")
        else:
            log_fn(f"{name}: {n_splits}-fold stratified cross-validation...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = cross_validate(estimator, X_train, y_train, cv=cv, scoring=ev.SCORING,
                                 error_score=np.nan, return_train_score=False)
        summary = ev.summarize_cv(raw)
        cv_results[name] = summary
        if summary["failed_folds"]:
            log_fn(f"  WARNING: {summary['failed_folds']} fold(s) failed for {name}.")
        m = summary[selection_metric]
        log_fn(f"  {selection_metric} = {m['mean'] if m['mean'] is None else round(m['mean'], 4)} "
               f"(+/- {m['std'] if m['std'] is None else round(m['std'], 4)})")
        # Final fit on the full training split (never on test data)
        try:
            estimator.fit(X_train, y_train)
            fitted[name] = estimator.best_estimator_ if tune else estimator
            best_params[name] = estimator.best_params_ if tune else {}
        except Exception as exc:
            log_fn(f"  Final fit failed for {name}: {exc}")
            cv_results[name]["final_fit_failed"] = True
        done += 1
        progress_fn(done / total_steps)

    # 2) Model selection by the predefined CV metric only
    candidates = [(n, r[selection_metric]["mean"], r[selection_metric]["std"]) for n, r in cv_results.items()
                  if r[selection_metric]["mean"] is not None and n in fitted]
    if not candidates:
        raise TrainingError("No model produced a valid cross-validation score.")
    candidates.sort(key=lambda t: (-t[1], t[2] if t[2] is not None else 0))
    best_name = candidates[0][0]
    best_pipe = fitted[best_name]
    log_fn(f"Selected '{best_name}' by mean CV {selection_metric} ({candidates[0][1]:.4f}).")

    # 3) One-time evaluation on the held-out test set
    proba_test = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics = ev.compute_metrics(y_test, proba_test, threshold)
    curves = ev.compute_curves(y_test, proba_test)
    log_fn(f"Held-out test: recall={test_metrics['recall']:.3f}, F1={test_metrics['f1']:.3f}, "
           f"ROC-AUC={test_metrics['roc_auc']:.3f}, accuracy={test_metrics['accuracy']:.3f}.")
    done += 1
    progress_fn(done / total_steps)

    # 4) Out-of-fold probabilities on TRAIN data for an honest threshold sweep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = cross_val_predict(clone(best_pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    sweep = ev.threshold_sweep(y_train, oof)
    done += 1
    progress_fn(done / total_steps)

    importance = _feature_importance(fitted, best_name, best_pipe, X_test, y_test, num_cols, cat_cols, seed, log_fn)
    done += 1
    progress_fn(done / total_steps)

    return {
        "version": ARTIFACT_VERSION,
        "pipeline": best_pipe,
        "model_name": best_name,
        "best_params": best_params.get(best_name, {}),
        "schema": build_schema(X_train, num_cols, cat_cols),
        "cv_results": cv_results,
        "selection_metric": selection_metric,
        "test_metrics": test_metrics,
        "curves": curves,
        "threshold_sweep": sweep,
        "importance": importance,
        "threshold": float(threshold),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(seed),
        "params": {"n_splits": n_splits, "class_weight": bool(class_weight), "tune": bool(tune),
                   "test_size": test_size},
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "train_attrition_rate": float(y_train.mean()), "test_attrition_rate": float(y_test.mean())},
        "dataset_signature": dataset_signature(df),
        "sklearn_version": sklearn.__version__,
    }


# ---------------------------------------------------------------- persistence
def save_artifact(artifact: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(artifact, tmp)
    os.replace(tmp, path)  # atomic: readers never see a half-written file


def load_artifact(path: Path, expected_signature: str | None = None) -> dict:
    """Load an artifact produced by this app. Only ever called with our own models/ path."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"No trained model found at {path.name}.")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise ArtifactError(f"Model artifact is corrupted or unreadable: {exc}") from exc
    if not isinstance(artifact, dict) or not REQUIRED_KEYS.issubset(artifact):
        raise ArtifactError("Model artifact has an unexpected structure.")
    # Check the public artifact contract before any route consumes nested values. This
    # catches truncated/hand-edited artifacts cleanly instead of failing later with a
    # confusing KeyError in a dashboard endpoint.
    expected_payload = {"best_params", "curves", "importance", "params", "split"}
    if not expected_payload.issubset(artifact):
        raise ArtifactError("Model artifact is incomplete (missing evaluation or schema metadata).")
    if not hasattr(artifact.get("pipeline"), "predict_proba"):
        raise ArtifactError("Model artifact does not contain a probability-capable pipeline.")
    if not isinstance(artifact.get("schema"), dict) or not isinstance(artifact["schema"].get("features"), list):
        raise ArtifactError("Model artifact has an invalid feature schema.")
    if not isinstance(artifact.get("cv_results"), dict) or not isinstance(artifact.get("test_metrics"), dict):
        raise ArtifactError("Model artifact has invalid evaluation results.")
    if artifact["version"] != ARTIFACT_VERSION:
        raise ArtifactError("Model artifact version is outdated.")
    if artifact["sklearn_version"] != sklearn.__version__:
        raise ArtifactError("Model was trained with a different scikit-learn version.")
    if expected_signature and artifact["dataset_signature"] != expected_signature:
        raise ArtifactError("Dataset changed since the model was trained.")
    return artifact
