"""Shared fixtures. The synthetic frame only mimics the *schema* of the IBM dataset for isolated
unit tests; the application itself always uses the real downloaded dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402


def make_synthetic_hr(n=400, seed=0):
    rng = np.random.default_rng(seed)
    overtime = rng.choice(["Yes", "No"], n, p=[0.28, 0.72])
    income = rng.integers(1000, 20000, n)
    sat = rng.integers(1, 5, n)
    logit = -2.0 + 1.4 * (overtime == "Yes") - 0.00008 * income - 0.3 * (sat - 2.5)
    left = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "Age": rng.integers(18, 60, n),
        "Attrition": np.where(left, "Yes", "No"),
        "BusinessTravel": rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n),
        "Department": rng.choice(["Sales", "Research & Development", "Human Resources"], n, p=[.3, .6, .1]),
        "DistanceFromHome": rng.integers(1, 30, n),
        "Education": rng.integers(1, 6, n),
        "EducationField": rng.choice(["Life Sciences", "Medical", "Marketing", "Other"], n),
        "EmployeeCount": 1,
        "EmployeeNumber": np.arange(1, n + 1),
        "EnvironmentSatisfaction": rng.integers(1, 5, n),
        "Gender": rng.choice(["Male", "Female"], n),
        "JobInvolvement": rng.integers(1, 5, n),
        "JobLevel": rng.integers(1, 6, n),
        "JobRole": rng.choice(["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"], n),
        "JobSatisfaction": sat,
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
        "MonthlyIncome": income,
        "Over18": "Y",
        "OverTime": overtime,
        "StandardHours": 80,
        "StockOptionLevel": rng.integers(0, 4, n),
        "WorkLifeBalance": rng.integers(1, 5, n),
        "YearsAtCompany": rng.integers(0, 30, n),
        "YearsInCurrentRole": rng.integers(0, 15, n),
        "YearsSinceLastPromotion": rng.integers(0, 15, n),
    })
    df.loc[3, "MonthlyIncome"] = np.nan  # a missing value to exercise imputation
    return df


@pytest.fixture(scope="session")
def hr_df():
    return make_synthetic_hr()


@pytest.fixture()
def service_app(tmp_path, hr_df):
    info = {"filename": "synthetic.csv", "source": "test", "rows": len(hr_df), "columns": hr_df.shape[1],
            "column_names": list(hr_df.columns),
            "validation": {"missing_total": 1, "duplicate_rows": 0, "missing_by_column": {}, "missing_expected_columns": []}}
    app = create_app({"TESTING": True, "MODEL_DIR": tmp_path, "DATA_DIR": tmp_path, "BOOTSTRAP_ON_START": False,
                      "AUTO_TRAIN": False}, dataset_provider=lambda: (hr_df.copy(), info))
    return app


@pytest.fixture()
def trained_app(service_app):
    svc = service_app.extensions["service"]
    assert svc.start_training({"n_splits": 3}, background=False)
    assert svc.train_state["status"] == "completed", svc.train_state["error"]
    return service_app
