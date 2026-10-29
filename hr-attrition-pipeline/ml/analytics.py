"""Descriptive analytics and the dataset explorer. Everything is computed from the real dataframe."""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

HIDDEN_COLUMNS = ["EmployeeNumber"]  # real identifier: never returned by the API
FILTER_COLUMNS = {"department": "Department", "gender": "Gender", "job_role": "JobRole",
                  "overtime": "OverTime", "attrition": "Attrition"}
LIKERT = {
    "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "EnvironmentSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
}


def jsonable(obj):
    """Recursively convert numpy / pandas values to JSON-safe Python values (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    for key, value in (filters or {}).items():
        col = FILTER_COLUMNS.get(key)
        if col and value and col in df.columns:
            df = df[df[col].astype(str) == str(value)]
    return df


def _rates(frame: pd.DataFrame, key: pd.Series, order=None, label_map=None) -> dict:
    left = (frame["Attrition"] == "Yes")
    g = pd.DataFrame({"k": key, "left": left}).dropna(subset=["k"]).groupby("k", observed=True)["left"]
    agg = g.agg(["size", "sum"])
    if order is not None:
        agg = agg.reindex([o for o in order if o in agg.index])
    agg = agg[agg["size"] > 0]
    labels = [label_map.get(i, str(i)) if label_map else str(i) for i in agg.index]
    total, left_n = agg["size"].astype(int).tolist(), agg["sum"].astype(int).tolist()
    return {"labels": labels, "total": total, "left": left_n, "stayed": [t - l for t, l in zip(total, left_n)],
            "rate": [round(100 * l / t, 2) if t else 0 for l, t in zip(left_n, total)]}


def _by_category(df, col, sort_rate=False):
    if col not in df.columns:
        return None
    r = _rates(df, df[col])
    if sort_rate:
        idx = sorted(range(len(r["labels"])), key=lambda i: -r["rate"][i])
        r = {k: [v[i] for i in idx] for k, v in r.items()}
    return r


def _by_bins(df, col, bins, labels):
    if col not in df.columns:
        return None
    return _rates(df, pd.cut(df[col], bins=bins, labels=labels, right=True, include_lowest=True), order=labels)


def _by_likert(df, col):
    if col not in df.columns:
        return None
    labels = LIKERT[col]
    r = _rates(df, df[col], order=sorted(df[col].dropna().unique()), label_map=labels)
    return r


def attrition_insights(df: pd.DataFrame, filters: dict | None = None) -> dict:
    data = apply_filters(df, filters)
    if data.empty:
        return {"empty": True, "message": "No employees match the selected filters.", "n": 0}
    n, left = len(data), int((data["Attrition"] == "Yes").sum())
    out = {
        "empty": False, "n": n,
        "overall": {"total": n, "left": left, "stayed": n - left, "rate": round(100 * left / n, 2)},
        "by_department": _by_category(data, "Department", True),
        "by_job_role": _by_category(data, "JobRole", True),
        "by_overtime": _by_category(data, "OverTime"),
        "by_business_travel": _by_category(data, "BusinessTravel", True),
        "by_age_group": _by_bins(data, "Age", [0, 25, 35, 45, 55, 120], ["<=25", "26-35", "36-45", "46-55", "56+"]),
        "by_years_at_company": _by_bins(data, "YearsAtCompany", [-1, 1, 3, 5, 10, 20, 100],
                                        ["0-1", "2-3", "4-5", "6-10", "11-20", "21+"]),
        "by_income_band": _by_bins(data, "MonthlyIncome", [0, 3000, 5000, 8000, 12000, 1e9],
                                   ["<3k", "3k-5k", "5k-8k", "8k-12k", "12k+"]),
        "by_job_satisfaction": _by_likert(data, "JobSatisfaction"),
        "by_work_life_balance": _by_likert(data, "WorkLifeBalance"),
    }
    if "MonthlyIncome" in data.columns:
        grp = data.groupby("Attrition")["MonthlyIncome"]
        out["income_comparison"] = {
            "labels": [g for g in ["No", "Yes"] if g in grp.groups],
            "mean": [round(float(grp.mean()[g]), 1) for g in ["No", "Yes"] if g in grp.groups],
            "median": [round(float(grp.median()[g]), 1) for g in ["No", "Yes"] if g in grp.groups],
        }
    return jsonable(out)


def overview(df: pd.DataFrame, model_artifact: dict | None) -> dict:
    n = len(df)
    left = int((df["Attrition"] == "Yes").sum())
    ins = attrition_insights(df)
    cv = None
    if model_artifact:
        best = model_artifact["cv_results"][model_artifact["model_name"]]
        m = model_artifact["selection_metric"]
        cv = {"model": model_artifact["model_name"], "metric": m, "mean": best[m]["mean"], "std": best[m]["std"]}
    return jsonable({
        "total": n, "left": left, "stayed": n - left, "rate": round(100 * left / n, 2),
        "n_features": int(df.shape[1] - 1), "cv": cv,
        "distribution": {"labels": ["Stayed", "Left"], "values": [n - left, left]},
        "department": ins["by_department"], "job_satisfaction": ins["by_job_satisfaction"],
        "overtime": ins["by_overtime"], "income": ins.get("income_comparison"), "age": ins["by_age_group"],
    })


# ------------------------------------------------------------------ explorer
def public_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in HIDDEN_COLUMNS if c in df.columns])


def column_summary(df: pd.DataFrame) -> dict:
    pub = public_frame(df)
    cols, numeric, categorical = [], {}, {}
    for c in pub.columns:
        is_num = pd.api.types.is_numeric_dtype(pub[c])
        cols.append({"name": c, "dtype": str(pub[c].dtype), "kind": "numeric" if is_num else "categorical",
                     "missing": int(pub[c].isna().sum()), "unique": int(pub[c].nunique(dropna=True))})
        if is_num:
            d = pub[c].describe()
            numeric[c] = {k: d[k] for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}
        else:
            vc = pub[c].value_counts().head(12)
            categorical[c] = {"top": vc.index.astype(str).tolist(), "counts": vc.tolist(),
                              "unique": int(pub[c].nunique())}
    return jsonable({"columns": cols, "numeric": numeric, "categorical": categorical})


def dataset_summary(df: pd.DataFrame, info: dict) -> dict:
    pub = public_frame(df)
    vc = df["Attrition"].value_counts()
    dup_full = df.duplicated()
    dup_content = pub.duplicated()
    return jsonable({
        "filename": info["filename"], "source": info["source"], "rows": info["rows"],
        "columns": info["columns"], "column_names": [c for c in info["column_names"] if c not in HIDDEN_COLUMNS],
        "validation": info["validation"],
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": {"full_rows": int(dup_full.sum()), "ignoring_identifier": int(dup_content.sum())},
        "target": {"labels": ["No", "Yes"], "counts": [int(vc.get("No", 0)), int(vc.get("Yes", 0))],
                   "rate": round(100 * float(vc.get("Yes", 0)) / len(df), 2)},
        "filter_options": {k: sorted(map(str, df[c].dropna().unique())) for k, c in FILTER_COLUMNS.items() if c in df.columns},
        "note": "EmployeeNumber is hidden from the explorer and from downloads.",
    })


def duplicates(df: pd.DataFrame, limit: int = 50) -> dict:
    pub = public_frame(df)
    mask = pub.duplicated(keep=False)
    return jsonable({"count": int(mask.sum()), "columns": list(pub.columns),
                     "rows": pub[mask].head(limit).to_dict(orient="records")})


def preview(df: pd.DataFrame, page=1, page_size=15, search="", sort_by=None, sort_dir="asc",
            columns=None, filters=None, max_page_size=100) -> dict:
    pub = public_frame(df)
    if sort_dir not in ("asc", "desc"):
        raise ValueError("sort_dir must be 'asc' or 'desc'")
    if sort_by and sort_by not in pub.columns:
        raise ValueError(f"Unknown sort column '{sort_by}'")
    if columns:
        bad = [c for c in columns if c not in pub.columns]
        if bad:
            raise ValueError(f"Unknown column(s): {', '.join(bad)}")
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be positive")
    page_size = min(page_size, max_page_size)
    frame = apply_filters(pub, filters)
    if search:
        s = search.strip().casefold()
        mask = frame.astype(str).apply(lambda col: col.str.casefold().str.contains(s, regex=False)).any(axis=1)
        frame = frame[mask]
    if sort_by:
        frame = frame.sort_values(sort_by, ascending=(sort_dir == "asc"), kind="mergesort")
    total = len(frame)
    pages = max(1, math.ceil(total / page_size))
    page = min(page, pages)
    chunk = frame.iloc[(page - 1) * page_size: page * page_size]
    shown = columns or list(pub.columns)
    return jsonable({"columns": shown, "rows": chunk[shown].to_dict(orient="records"), "total": total,
                     "page": page, "pages": pages, "page_size": page_size})


def column_chart(df: pd.DataFrame, column: str, bins: int = 12) -> dict:
    pub = public_frame(df)
    if column not in pub.columns:
        raise ValueError(f"Unknown column '{column}'")
    if not 2 <= bins <= 50:
        raise ValueError("bins must be between 2 and 50")
    s = pub[column]
    left = pub["Attrition"] == "Yes"
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
        cut = pd.cut(s, bins=bins)
        labels = [f"{iv.left:.0f}-{iv.right:.0f}" if abs(iv.right) >= 20 else f"{iv.left:.1f}-{iv.right:.1f}"
                  for iv in cut.cat.categories]
        key = cut.cat.rename_categories(labels)
    else:
        key = s.astype(str)
    g = pd.DataFrame({"k": key, "left": left}).groupby("k", observed=True)["left"].agg(["size", "sum"])
    if not pd.api.types.is_numeric_dtype(s) or s.nunique() <= 12:
        try:
            g = g.loc[sorted(g.index, key=lambda v: float(v))]
        except ValueError:
            g = g.sort_values("size", ascending=False).head(15)
    return jsonable({"column": column, "labels": [str(i) for i in g.index], "total": g["size"].tolist(),
                     "left": g["sum"].tolist(), "stayed": (g["size"] - g["sum"]).tolist()})


def filtered_csv(df: pd.DataFrame, filters=None, search="") -> str:
    frame = apply_filters(public_frame(df), filters)
    if search:
        s = search.strip().casefold()
        frame = frame[frame.astype(str).apply(lambda c: c.str.casefold().str.contains(s, regex=False)).any(axis=1)]
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    return buf.getvalue()
