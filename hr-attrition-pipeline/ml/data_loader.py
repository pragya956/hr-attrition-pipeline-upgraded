"""Dataset acquisition (KaggleHub), CSV discovery, validation and caching."""
from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Callable

import pandas as pd

try:  # kagglehub is optional at import time so unit tests can run without it
    import kagglehub
except ImportError:  # pragma: no cover
    kagglehub = None

log = logging.getLogger(__name__)

TARGET = "Attrition"
KAGGLE_HANDLE = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
# Columns the dashboards and the prediction form rely on. Missing ones are reported as warnings.
EXPECTED_COLUMNS = [
    "Age", "BusinessTravel", "Department", "DistanceFromHome", "Education", "EducationField",
    "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "OverTime", "StockOptionLevel", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
]


class DatasetError(RuntimeError):
    """Raised for any problem acquiring or validating the dataset."""


def download_dataset() -> Path:
    """Download the IBM HR Analytics dataset with KaggleHub and return its directory."""
    if kagglehub is None:
        raise DatasetError("The 'kagglehub' package is not installed. Run: pip install kagglehub")
    try:
        # Download latest version
        path = kagglehub.dataset_download(
            "pavansubhasht/ibm-hr-analytics-attrition-dataset"
        )
        print("Path to dataset files:", path)
    except Exception as exc:  # network, auth, rate limit, ...
        raise DatasetError(
            f"KaggleHub download failed: {exc}. Check your internet connection and, if needed, "
            "Kaggle credentials (KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json)."
        ) from exc
    return Path(path)


def find_csv(directory: Path | str) -> Path:
    """Locate the CSV that contains the target column (no hardcoded filename)."""
    directory = Path(directory)
    if not directory.exists():
        raise DatasetError(f"Dataset directory does not exist: {directory}")
    candidates = sorted(directory.rglob("*.csv"))
    if not candidates:
        raise DatasetError(f"No CSV file found inside {directory}")
    for candidate in candidates:
        try:
            header = pd.read_csv(candidate, nrows=0, encoding="utf-8-sig").columns
        except Exception as exc:
            log.warning("Skipping unreadable CSV %s: %s", candidate, exc)
            continue
        if TARGET in header:
            return candidate
    raise DatasetError(f"No CSV in {directory} contains the target column '{TARGET}'")


def validate_schema(df: pd.DataFrame) -> dict:
    """Validate the dataframe and return a report. Raises DatasetError on fatal problems."""
    if df is None or df.empty:
        raise DatasetError("The dataset is empty.")
    if TARGET not in df.columns:
        raise DatasetError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")
    labels = set(df[TARGET].dropna().unique())
    if not labels or not labels.issubset({"Yes", "No"}):
        raise DatasetError(f"'{TARGET}' must contain only 'Yes'/'No'; found {sorted(map(str, labels))}")
    if len(labels) < 2:
        raise DatasetError(f"'{TARGET}' has only one class; cannot train a classifier.")
    if df[TARGET].isna().any():
        raise DatasetError(f"'{TARGET}' contains missing values.")
    missing = df.isna().sum()
    missing_cols = {c: int(n) for c, n in missing.items() if n > 0}
    return {
        "target_found": True,
        "missing_total": int(missing.sum()),
        "missing_by_column": missing_cols,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_expected_columns": [c for c in EXPECTED_COLUMNS if c not in df.columns],
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise DatasetError(f"Could not read CSV {path}: {exc}") from exc


def load_dataset(
    data_dir: Path | str,
    local_csv_name: str = "ibm_hr_attrition.csv",
    downloader: Callable[[], Path] = download_dataset,
) -> tuple[pd.DataFrame, dict]:
    """Download (or reuse a previously downloaded copy), load and validate the dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    local_copy = data_dir / local_csv_name
    source = "kagglehub"
    try:
        csv_path = find_csv(downloader())
        # Keep an offline copy of the *real* downloaded file for later runs
        try:
            if csv_path.resolve() != local_copy.resolve():
                shutil.copyfile(csv_path, local_copy)
        except OSError as exc:
            log.warning("Could not cache dataset locally: %s", exc)
    except DatasetError as exc:
        if local_copy.exists():
            log.warning("%s -- falling back to the cached copy %s", exc, local_copy)
            csv_path, source = local_copy, "local cache of the Kaggle download"
        else:
            raise
    df = _read_csv(csv_path)
    report = validate_schema(df)
    info = {
        "filename": csv_path.name,
        "source": source,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "validation": report,
    }
    log.info("Loaded %s: %d rows x %d columns", csv_path.name, *df.shape)
    return df, info


class DatasetCache:
    """Thread-safe in-memory cache so the CSV is downloaded and parsed once."""

    def __init__(self, loader: Callable[[], tuple[pd.DataFrame, dict]]):
        self._loader = loader
        self._lock = threading.Lock()
        self.df: pd.DataFrame | None = None
        self.info: dict | None = None
        self.error: str | None = None

    def get(self, force: bool = False) -> tuple[pd.DataFrame, dict]:
        with self._lock:
            if self.df is None or force:
                try:
                    self.df, self.info = self._loader()
                    self.error = None
                except DatasetError as exc:
                    self.error = str(exc)
                    raise
            return self.df, self.info

    @property
    def state(self) -> str:
        if self.df is not None:
            return "ready"
        return "error" if self.error else "loading"
