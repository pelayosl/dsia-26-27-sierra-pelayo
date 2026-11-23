"""Tests of `infrastructure.model_store`."""

import json
import logging
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pytest
from pydantic_core import PydanticSerializationError

from energy_forecast.domain.errors import ModelNotFoundError
from energy_forecast.domain.features import build_training_frame
from energy_forecast.domain.models import ModelMetrics
from energy_forecast.domain.validator import validate_observations
from energy_forecast.infrastructure.model_store import JoblibModelStore, StoredModel
from tests.fakes import FakeModelStore

SAVED_AT = datetime(2026, 11, 7, 12, 0, 0, tzinfo=timezone.utc)
METRICS = {"hgb": ModelMetrics(mae=600.0, rmse=900.0, mape=2.2), "features": ("lag_24", "hour")}


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "models" / "model.joblib", tmp_path / "reports" / "metrics.json"


@pytest.fixture
def store(paths) -> JoblibModelStore:
    return JoblibModelStore(*paths, clock=lambda: SAVED_AT)


@pytest.fixture
def X(hourly_observations):
    observations, _ = validate_observations(hourly_observations)
    return build_training_frame(observations)[0]


def test_save_and_load_roundtrip(store, paths, small_trained_model, X):
    # Arrange
    model_path, metrics_path = paths

    # Act
    version = store.save(small_trained_model, "hgb", METRICS)
    loaded = store.load()

    # Assert
    assert version == "hgb-20261107T120000Z"
    assert model_path.exists() and metrics_path.exists()
    assert isinstance(loaded, StoredModel)
    assert loaded.version == version
    np.testing.assert_array_equal(loaded.model.predict(X), small_trained_model.predict(X))


def test_metrics_json_is_readable_and_matches_loaded_metrics(store, paths, small_trained_model):
    _, metrics_path = paths

    store.save(small_trained_model, "hgb", METRICS)
    report = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert report["model_version"] == "hgb-20261107T120000Z"
    assert report["saved_at"] == "2026-11-07T12:00:00Z"
    assert report["hgb"] == {"mae": 600.0, "rmse": 900.0, "mape": 2.2}  # Pydantic models become dicts
    assert report["features"] == ["lag_24", "hour"]  # tuples become lists
    assert store.load().metrics == report


def test_version_uses_utc_even_if_the_clock_is_local(paths, small_trained_model):
    madrid = timezone(timedelta(hours=1))
    store = JoblibModelStore(*paths, clock=lambda: datetime(2026, 11, 7, 13, 0, tzinfo=madrid))

    assert store.save(small_trained_model, "ridge", {}) == "ridge-20261107T120000Z"


def test_save_overwrites_the_previous_model(paths, small_trained_model):
    times = iter([SAVED_AT, SAVED_AT + timedelta(days=1)])
    store = JoblibModelStore(*paths, clock=lambda: next(times))

    store.save(small_trained_model, "ridge", {})
    store.save(small_trained_model, "hgb", {})

    assert store.load().version == "hgb-20261108T120000Z"
    assert not list(paths[0].parent.glob("*.tmp"))  # no temporary files left behind


def test_failed_save_keeps_the_previous_model(store, paths, small_trained_model):
    store.save(small_trained_model, "hgb", METRICS)

    with pytest.raises(PydanticSerializationError):
        store.save(small_trained_model, "broken", {"not_serialisable": object()})

    assert store.load().version == "hgb-20261107T120000Z"
    assert json.loads(paths[1].read_text(encoding="utf-8"))["model_version"] == "hgb-20261107T120000Z"


def test_load_without_model_raises(store):
    with pytest.raises(ModelNotFoundError, match="run `train` first"):
        store.load()


def test_load_corrupt_file_raises_model_not_found(store, paths):
    model_path, _ = paths
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"this is not a joblib file")

    with pytest.raises(ModelNotFoundError, match="could not be loaded"):
        store.load()


def test_load_file_from_another_program_raises_model_not_found(store, paths, small_trained_model):
    model_path, _ = paths
    model_path.parent.mkdir(parents=True)
    joblib.dump(small_trained_model, model_path)  # a bare model, without version or metrics

    with pytest.raises(ModelNotFoundError, match="not a model saved by `train`"):
        store.load()


def test_load_warns_when_sklearn_version_differs(store, paths, small_trained_model, caplog):
    model_path, _ = paths
    store.save(small_trained_model, "hgb", {})
    payload = joblib.load(model_path)
    joblib.dump({**payload, "sklearn_version": "0.0.1"}, model_path)

    with caplog.at_level(logging.WARNING):
        store.load()

    assert "scikit-learn 0.0.1" in caplog.text


# ---------------------------------------------------------------- contract shared with the fake


@pytest.fixture(params=["joblib", "fake"])
def any_store(request, store):
    return store if request.param == "joblib" else FakeModelStore()


def test_store_contract_empty_store_raises(any_store):
    with pytest.raises(ModelNotFoundError):
        any_store.load()


def test_store_contract_save_then_load(any_store, small_trained_model):
    version = any_store.save(small_trained_model, "hgb", METRICS)

    loaded = any_store.load()

    assert version.startswith("hgb-")
    assert loaded.version == version
    assert type(loaded.model) is type(small_trained_model)
    assert loaded.metrics["model_version"] == version
    assert loaded.metrics["hgb"] == {"mae": 600.0, "rmse": 900.0, "mape": 2.2}
