import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml import evaluate as ev
from ml.preprocessing import (build_model_pipelines, build_preprocessor, map_feature_names, prepare_xy,
                              select_feature_columns, split_column_types)
from ml.train import (ArtifactError, TrainingError, load_artifact, save_artifact, train_pipeline)


def test_prepare_xy_drops_identifiers_and_encodes_target(hr_df):
    X, y = prepare_xy(hr_df)
    for col in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours", "Attrition"]:
        assert col not in X.columns
    assert set(y.unique()) == {0, 1} and int(y.sum()) == int((hr_df.Attrition == "Yes").sum())


def test_column_types_are_dynamic(hr_df):
    X, _ = prepare_xy(hr_df)
    num, cat = split_column_types(X)
    assert "Age" in num and "Department" in cat and not set(num) & set(cat)


def test_constant_columns_dropped_from_train_split():
    X = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    assert select_feature_columns(X) == ["b"]


def test_preprocessor_imputes_and_handles_unseen_categories(hr_df):
    X, _ = prepare_xy(hr_df)
    num, cat = split_column_types(X)
    pre = build_preprocessor(num, cat, scale=True).fit(X)
    new = X.iloc[:2].copy()
    new["Department"] = "Never Seen Department"
    new.loc[new.index[0], "MonthlyIncome"] = np.nan
    out = pre.transform(new)
    assert not np.isnan(out).any()
    names = list(pre.get_feature_names_out())
    dept_cols = [i for i, n in enumerate(names) if n.startswith("cat__Department_")]
    assert out[0, dept_cols].sum() == 0  # unseen -> all zeros, no exception


def test_feature_name_mapping(hr_df):
    X, _ = prepare_xy(hr_df)
    num, cat = split_column_types(X)
    pre = build_preprocessor(num, cat, scale=False).fit(X)
    mapped = map_feature_names(pre.get_feature_names_out(), num, cat)
    originals = {m["original"] for m in mapped}
    assert originals == set(X.columns)
    assert any(m["original"] == "JobRole" and m["level"] == "Manager" for m in mapped)


def test_stratified_split_keeps_class_ratio(hr_df):
    X, y = prepare_xy(hr_df)
    _, _, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    assert abs(ytr.mean() - yte.mean()) < 0.02


def test_encoder_is_fitted_per_fold_only(hr_df):
    """A category present only in the validation fold must be unseen by that fold's encoder."""
    X, y = prepare_xy(hr_df)
    X = X.copy()
    X["EducationField"] = X["EducationField"].astype(object)
    X.iloc[0, X.columns.get_loc("EducationField")] = "RARE_ONLY_ONCE"
    num, cat = split_column_types(X)
    pipe = build_model_pipelines(num, cat)["Logistic Regression"]
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=1).split(X, y):
        if 0 in va:
            pipe.fit(X.iloc[tr], y.iloc[tr])
            enc = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
            idx = cat.index("EducationField")
            assert "RARE_ONLY_ONCE" not in enc.categories_[idx]
            pipe.predict_proba(X.iloc[va])  # still works


def test_metric_edge_cases():
    m = ev.compute_metrics([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert m["roc_auc"] is None and m["recall"] == 0.0
    m = ev.compute_metrics([0, 0, 0, 1], [0.1, 0.1, 0.1, 0.1])
    assert m["accuracy"] == 0.75 and m["recall"] == 0.0 and m["balanced_accuracy"] == 0.5


def test_train_pipeline_outputs(hr_df):
    logs = []
    art = train_pipeline(hr_df, n_splits=3, seed=7, log_fn=logs.append)
    assert art["model_name"] in art["cv_results"]
    for name, res in art["cv_results"].items():
        assert res["n_folds"] == 3 and res["failed_folds"] == 0
        assert 0 <= res["roc_auc"]["mean"] <= 1
    # selection used CV only: chosen model has the highest mean of the selection metric
    metric = art["selection_metric"]
    best = max(art["cv_results"], key=lambda n: art["cv_results"][n][metric]["mean"])
    assert best == art["model_name"]
    assert art["split"]["train_rows"] + art["split"]["test_rows"] == len(hr_df)
    assert art["test_metrics"]["n_samples"] == art["split"]["test_rows"]
    assert "permutation" in art["importance"] and any("Selected" in l for l in logs)


def test_training_is_reproducible(hr_df):
    a = train_pipeline(hr_df, n_splits=3, seed=11)
    b = train_pipeline(hr_df, n_splits=3, seed=11)
    assert a["test_metrics"]["roc_auc"] == pytest.approx(b["test_metrics"]["roc_auc"])
    assert a["model_name"] == b["model_name"]


def test_invalid_training_params(hr_df):
    with pytest.raises(TrainingError):
        train_pipeline(hr_df, n_splits=1)
    with pytest.raises(TrainingError):
        train_pipeline(hr_df, selection_metric="accuracy")


def test_artifact_roundtrip_and_failures(hr_df, tmp_path):
    art = train_pipeline(hr_df, n_splits=3)
    path = tmp_path / "a.joblib"
    save_artifact(art, path)
    loaded = load_artifact(path, expected_signature=art["dataset_signature"])
    X, _ = prepare_xy(hr_df)
    cols = art["schema"]["columns"]
    assert np.allclose(art["pipeline"].predict_proba(X[cols]), loaded["pipeline"].predict_proba(X[cols]))
    with pytest.raises(ArtifactError, match="Dataset changed"):
        load_artifact(path, expected_signature="different")
    with pytest.raises(ArtifactError, match="No trained model"):
        load_artifact(tmp_path / "missing.joblib")
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    with pytest.raises(ArtifactError, match="corrupted"):
        load_artifact(bad)
    import joblib
    joblib.dump({"foo": 1}, tmp_path / "weird.joblib")
    with pytest.raises(ArtifactError, match="structure"):
        load_artifact(tmp_path / "weird.joblib")
    incomplete = dict(art)
    incomplete.pop("curves")
    joblib.dump(incomplete, tmp_path / "incomplete.joblib")
    with pytest.raises(ArtifactError, match="incomplete"):
        load_artifact(tmp_path / "incomplete.joblib")
