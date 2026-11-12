"""Tests for `infrastructure.dataset_source` (no network: URLs are local `file://` or simulated)."""

from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest

from energy_forecast.domain.errors import DataLoadError
from energy_forecast.infrastructure import dataset_source
from energy_forecast.infrastructure.dataset_source import CsvDatasetSource


@pytest.fixture
def csv_path(tmp_path: Path, hourly_observations: pd.DataFrame) -> Path:
    """`hourly_observations` written as a CSV with the same format as dsia-dataset."""
    path = tmp_path / "energy_weather_es_hourly.csv"
    hourly_observations.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("as_type", [str, Path])
def test_reads_csv_from_local_path(csv_path, hourly_observations, as_type):
    # Arrange
    source = CsvDatasetSource(as_type(csv_path))

    # Act
    frame = source.read()

    # Assert
    pd.testing.assert_frame_equal(frame, hourly_observations)


def test_reads_csv_from_url(csv_path, hourly_observations):
    source = CsvDatasetSource(csv_path.as_uri())  # file:// URL: goes through the URL branch without network

    frame = source.read()

    pd.testing.assert_frame_equal(frame, hourly_observations)


def test_returns_data_without_validating(tmp_path, invalid_observations):
    path = tmp_path / "invalid.csv"
    invalid_observations.to_csv(path, index=False)

    frame = CsvDatasetSource(path).read()

    assert len(frame) == len(invalid_observations)
    assert frame["time_utc"].duplicated().any()


@pytest.mark.parametrize("as_type", [str, Path])
def test_missing_file_raises_data_load_error(tmp_path, as_type):
    source = CsvDatasetSource(as_type(tmp_path / "missing.csv"))

    with pytest.raises(DataLoadError, match="missing.csv"):
        source.read()


def test_empty_file_raises_data_load_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DataLoadError, match="empty.csv"):
        CsvDatasetSource(path).read()


def test_malformed_csv_raises_data_load_error(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")

    with pytest.raises(DataLoadError):
        CsvDatasetSource(path).read()


def test_unreachable_url_raises_data_load_error(monkeypatch):
    def fail_download(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(dataset_source.pd, "read_csv", fail_download)

    with pytest.raises(DataLoadError):
        CsvDatasetSource("https://example.org/energy_weather_es_hourly.csv").read()
