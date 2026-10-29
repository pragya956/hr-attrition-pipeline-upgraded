import pytest

from ml.predict import ValidationError, predict_profile, validate_profile
from ml.train import train_pipeline


@pytest.fixture(scope="module")
def art():
    from tests.conftest import make_synthetic_hr
    return train_pipeline(make_synthetic_hr(), n_splits=3)


GOOD = {"Age": 30, "Department": "Sales", "MonthlyIncome": 4000, "OverTime": "Yes", "JobSatisfaction": 2}


def test_valid_prediction_has_probability_and_class(art):
    out = predict_profile(art, GOOD)
    assert 0 <= out["probability"] <= 1
    assert out["prediction"] in ("Likely to leave", "Likely to stay")
    assert "not suitable" in out["disclaimer"]


def test_threshold_changes_class_without_retraining(art):
    low = predict_profile(art, GOOD, threshold=0.01)
    high = predict_profile(art, GOOD, threshold=0.99)
    assert low["probability"] == high["probability"]
    assert low["predicted_class"] == 1 and high["predicted_class"] == 0


def test_unseen_category_warns_but_predicts(art):
    out = predict_profile(art, {**GOOD, "Department": "Underwater Basket Weaving"})
    assert any("unseen" in w for w in out["warnings"])


@pytest.mark.parametrize("bad,field", [
    ({"Age": "abc"}, "Age"), ({"JobSatisfaction": 9}, "JobSatisfaction"),
    ({"MonthlyIncome": -5}, "MonthlyIncome"), ({"Age": 30.5}, "Age"),
    ({"Age": float("nan")}, "Age"), ({"JobSatisfaction": 0}, "JobSatisfaction"),
])
def test_invalid_values_rejected(art, bad, field):
    with pytest.raises(ValidationError) as e:
        validate_profile(art["schema"], bad)
    assert field in e.value.errors


def test_unknown_field_rejected_as_incompatible_schema(art):
    with pytest.raises(ValidationError):
        validate_profile(art["schema"], {"NotAFeature": 1})
    with pytest.raises(ValidationError):
        validate_profile(art["schema"], ["not", "a", "dict"])


def test_missing_fields_use_training_defaults(art):
    df, row, _ = validate_profile(art["schema"], {})
    assert list(df.columns) == art["schema"]["columns"] and len(df) == 1
