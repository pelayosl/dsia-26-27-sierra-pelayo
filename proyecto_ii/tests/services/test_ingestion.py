"""Tests for `IngestionService` with fakes (no database, no files)."""

import logging

import pytest

from energy_forecast.domain.errors import DataLoadError, SchemaError
from energy_forecast.services.ingestion import IngestionService
from tests.fakes import FakeDatasetSource, InMemoryObservationRepository, InMemoryPipelineRunRepository


def build_service(source: FakeDatasetSource):
    observations = InMemoryObservationRepository()
    runs = InMemoryPipelineRunRepository()
    return IngestionService(source, observations, runs), observations, runs


def test_ingestion_writes_valid_rows_and_reports(invalid_observations):
    # Arrange: 3 rows are rejected (duplicated time x2 + invalid flag)
    service, observations, runs = build_service(FakeDatasetSource(invalid_observations))

    # Act
    report = service.run()

    # Assert
    assert report.rows_in == len(invalid_observations)
    assert report.rows_rejected == 3
    assert report.rows_valid == len(invalid_observations) - 3
    assert report.rows_written == report.rows_valid
    assert len(observations.range()) == report.rows_valid
    assert runs.get("run-1") == {"run_id": "run-1", "kind": "ingest", "status": "ok", "rows": report.rows_written, "error": None}


def test_ingestion_is_idempotent(hourly_observations):
    service, observations, _ = build_service(FakeDatasetSource(hourly_observations))

    service.run()
    service.run()

    assert len(observations.range()) == len(hourly_observations)


def test_ingestion_records_failed_run_on_schema_error(hourly_observations):
    service, observations, runs = build_service(FakeDatasetSource(hourly_observations.drop(columns=["load_mw"])))

    with pytest.raises(SchemaError):
        service.run()

    run = runs.get("run-1")
    assert run["status"] == "error"
    assert run["error"].startswith("SchemaError")
    assert observations.range().empty


def test_ingestion_records_failed_run_on_load_error():
    service, _, runs = build_service(FakeDatasetSource(error=DataLoadError("File not found: x.csv")))

    with pytest.raises(DataLoadError):
        service.run()

    assert runs.get("run-1")["status"] == "error"
    assert "x.csv" in runs.get("run-1")["error"]


def test_ingestion_logs_rejected_rows_by_reason(invalid_observations, caplog):
    service, _, _ = build_service(FakeDatasetSource(invalid_observations))

    with caplog.at_level(logging.INFO, logger="energy_forecast"):
        service.run()

    assert "read=384 valid=381 rejected=3 written=381" in caplog.text
    assert "'duplicated_time_utc': 2" in caplog.text
    assert "'invalid_flag': 1" in caplog.text
