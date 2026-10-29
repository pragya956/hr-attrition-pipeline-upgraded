from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from ml import data_loader as dl
from tests.conftest import make_synthetic_hr


def test_find_csv_discovers_file_without_hardcoded_name(tmp_path, hr_df):
    sub = tmp_path / "nested"
    sub.mkdir()
    pd.DataFrame({"x": [1]}).to_csv(sub / "other.csv", index=False)
    hr_df.to_csv(sub / "whatever_name.csv", index=False)
    assert dl.find_csv(tmp_path).name == "whatever_name.csv"


def test_find_csv_errors(tmp_path):
    with pytest.raises(dl.DatasetError, match="No CSV"):
        dl.find_csv(tmp_path)
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(dl.DatasetError, match="Attrition"):
        dl.find_csv(tmp_path)
    with pytest.raises(dl.DatasetError):
        dl.find_csv(tmp_path / "missing")


def test_validate_schema_reports_missing_and_duplicates(hr_df):
    report = dl.validate_schema(pd.concat([hr_df, hr_df.iloc[[0]]]))
    assert report["duplicate_rows"] == 1
    assert report["missing_total"] >= 1


def test_validate_schema_rejects_bad_target(hr_df):
    with pytest.raises(dl.DatasetError):
        dl.validate_schema(hr_df.drop(columns="Attrition"))
    bad = hr_df.copy()
    bad["Attrition"] = "Maybe"
    with pytest.raises(dl.DatasetError):
        dl.validate_schema(bad)
    one = hr_df.copy()
    one["Attrition"] = "No"
    with pytest.raises(dl.DatasetError, match="one class"):
        dl.validate_schema(one)


def test_download_uses_kagglehub_mock(tmp_path):
    fake = mock.Mock()
    fake.dataset_download.return_value = str(tmp_path)
    with mock.patch.object(dl, "kagglehub", fake):
        assert dl.download_dataset() == tmp_path
    fake.dataset_download.assert_called_once_with("pavansubhasht/ibm-hr-analytics-attrition-dataset")


def test_download_failure_gives_clear_error():
    fake = mock.Mock()
    fake.dataset_download.side_effect = RuntimeError("offline")
    with mock.patch.object(dl, "kagglehub", fake), pytest.raises(dl.DatasetError, match="offline"):
        dl.download_dataset()


def test_load_dataset_copies_then_falls_back_to_cache(tmp_path, hr_df):
    src, data = tmp_path / "src", tmp_path / "data"
    src.mkdir()
    hr_df.to_csv(src / "real.csv", index=False)
    df, info = dl.load_dataset(data, downloader=lambda: src)
    assert info["rows"] == len(hr_df) and info["filename"] == "real.csv"
    assert (data / "ibm_hr_attrition.csv").exists()

    def failing():
        raise dl.DatasetError("no network")
    df2, info2 = dl.load_dataset(data, downloader=failing)
    assert "cache" in info2["source"] and len(df2) == len(hr_df)


def test_load_dataset_fails_without_cache(tmp_path):
    def failing():
        raise dl.DatasetError("no network")
    with pytest.raises(dl.DatasetError):
        dl.load_dataset(tmp_path, downloader=failing)
