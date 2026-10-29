"""Feature preparation and the leakage-free preprocessing ColumnTransformer.

Everything that *learns* from data (imputers, scaler, one-hot encoder) lives inside the
sklearn Pipeline, so it is fitted on the training part of each CV fold only.
"""
from __future__ import annotations

import inspect

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Attrition"
LABEL_MAP = {"No": 0, "Yes": 1}
# Identifiers / known constant columns of the IBM dataset: never used as predictors.
EXCLUDED_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw dataframe into features and a 0/1 target. No statistics are learned here."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")
    y = df[TARGET].map(LABEL_MAP)
    if y.isna().any():
        raise ValueError(f"'{TARGET}' must only contain 'Yes'/'No'.")
    X = df.drop(columns=[TARGET] + [c for c in EXCLUDED_COLUMNS if c in df.columns])
    return X, y.astype(int)


def select_feature_columns(X_train: pd.DataFrame) -> list[str]:
    """Drop columns that are constant *in the training split* (no information)."""
    return [c for c in X_train.columns if X_train[c].nunique(dropna=True) > 1]


def split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Determine numerical vs categorical columns dynamically from the training features."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return num, cat


def build_preprocessor(num_cols: list[str], cat_cols: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:  # scaling matters for logistic regression, not for tree ensembles
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # unseen categories at prediction time become an all-zero block instead of raising
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", Pipeline(num_steps), num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _hgb_supports_class_weight() -> bool:
    return "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters


def build_model_pipelines(num_cols, cat_cols, seed: int = 42, class_weight: bool = True) -> dict[str, Pipeline]:
    """One complete preprocessing + classifier pipeline per model."""
    cw = "balanced" if class_weight else None
    hgb_kwargs = {"random_state": seed, "max_iter": 200, "learning_rate": 0.06}
    if class_weight and _hgb_supports_class_weight():
        hgb_kwargs["class_weight"] = "balanced"
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor(num_cols, cat_cols, scale=True)),
            ("model", LogisticRegression(max_iter=3000, class_weight=cw, random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", build_preprocessor(num_cols, cat_cols, scale=False)),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed,
                class_weight="balanced_subsample" if class_weight else None)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocess", build_preprocessor(num_cols, cat_cols, scale=False)),
            ("model", HistGradientBoostingClassifier(**hgb_kwargs)),
        ]),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "Random Forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 6, 12],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "Gradient Boosting": {
        "model__learning_rate": [0.03, 0.06, 0.1],
        "model__max_depth": [None, 3, 5],
        "model__max_iter": [100, 200],
    },
}


def map_feature_names(transformed_names, num_cols, cat_cols) -> list[dict]:
    """Map 'num__Age' / 'cat__Department_Sales' back to readable original features."""
    out = []
    cats_by_len = sorted(cat_cols, key=len, reverse=True)
    for name in transformed_names:
        if name.startswith("num__"):
            out.append({"name": name, "original": name[5:], "level": None})
            continue
        rest = name[5:] if name.startswith("cat__") else name
        match = next((c for c in cats_by_len if rest.startswith(c + "_")), None)
        if match:
            out.append({"name": name, "original": match, "level": rest[len(match) + 1:]})
        else:
            out.append({"name": name, "original": rest, "level": None})
    return out
