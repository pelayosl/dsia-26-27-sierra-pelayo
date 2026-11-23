"""Tests of `TrainingService` with fakes (no database or files)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from energy_forecast.domain import modeling
from energy_forecast.domain.errors import InsufficientHistoryError
from energy_forecast.domain.features import FEATURE_COLUMNS
from energy_forecast.domain.modeling import TRAIN_END, VALIDATION_END
from energy_forecast.domain.models import TrainingResult
from energy_forecast.domain.validator import validate_observations
from energy_forecast.services import training
from energy_forecast.services.training import TrainingService
from tests.fakes import FakeModelStore, InMemoryObservationRepository, InMemoryPipelineRunRepository


class OffsetModel(BaseEstimator, RegressorMixin):
    """Predicts `lag_24 + offset`: the larger the offset, the worse the model. Records the rows of each fit."""

    fitted_rows: list[int] = []  # shared by the clones the service creates

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OffsetModel":
        OffsetModel.fitted_rows.append(len(X))
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_24"].to_numpy(dtype=float) + self.offset


@pytest.fixture
def stub_candidates(monkeypatch):
    """Candidates with a known order: `good` (offset 0) < `medium` < `bad` in error."""
    OffsetModel.fitted_rows = []
    candidates = {"bad": OffsetModel(3_000.0), "good": OffsetModel(0.0), "medium": OffsetModel(800.0)}
    monkeypatch.setattr(training, "build_candidates", lambda: candidates)
    return candidates


@pytest.fixture
def observations_repository(training_span_observations) -> InMemoryObservationRepository:
    repository = InMemoryObservationRepository()
    repository.upsert(validate_observations(training_span_observations)[0])
    return repository


@pytest.fixture
def store() -> FakeModelStore:
    return FakeModelStore()


@pytest.fixture
def runs() -> InMemoryPipelineRunRepository:
    return InMemoryPipelineRunRepository()


@pytest.fixture
def service(observations_repository, store, runs) -> TrainingService:
    return TrainingService(observations_repository, store, runs)


def test_training_reports_metrics_for_every_candidate(service, stub_candidates):
    result = service.train()

    assert isinstance(result, TrainingResult)
    assert set(result.metrics) == set(result.validation_metrics) == {"bad", "good", "medium"}
    assert result.metrics["good"].mae < result.metrics["medium"].mae < result.metrics["bad"].mae


def test_training_saves_best_model(service, store, stub_candidates):
    result = service.train()

    stored = store.load()
    assert result.best_model == "good"
    assert stored.model.offset == 0.0
    assert result.model_version == stored.version == "good-test-v1"


def test_best_model_is_chosen_by_validation_not_test(service, stub_candidates, monkeypatch):
    # `sneaky` is worse on validation but better on test: it must not be chosen
    class SneakyModel(OffsetModel):
        def predict(self, X):
            in_test = X.index > VALIDATION_END
            return X["lag_24"].to_numpy(dtype=float) + np.where(in_test, 0.0, 5_000.0)

    candidates = {"good": OffsetModel(400.0), "sneaky": SneakyModel()}
    monkeypatch.setattr(training, "build_candidates", lambda: candidates)

    result = service.train()

    assert result.metrics["sneaky"].mae < result.metrics["good"].mae
    assert result.best_model == "good"


def test_candidates_are_fitted_on_train_then_on_train_plus_validation(service, stub_candidates):
    result = service.train()

    train_rows, train_val_rows = OffsetModel.fitted_rows[0], OffsetModel.fitted_rows[1]
    assert len(OffsetModel.fitted_rows) == 2 * len(stub_candidates)
    assert train_rows < train_val_rows
    assert result.train_end <= VALIDATION_END < result.test_start


def test_candidates_passed_to_the_service_are_not_modified(service, stub_candidates):
    service.train()

    assert not any(hasattr(model, "is_fitted_") for model in stub_candidates.values())  # the service fits clones


def test_training_result_describes_the_data(service, stub_candidates):
    result = service.train()

    assert result.features == tuple(FEATURE_COLUMNS)
    assert result.train_start < TRAIN_END < result.train_end < result.test_start <= result.test_end
    assert sorted(result.mae_by_hour) == list(range(24))


def test_saved_metrics_match_the_result(service, store, stub_candidates):
    result = service.train()

    saved = store.load().metrics
    assert saved["model_version"] == result.model_version
    assert saved["best_model"] == "good"
    assert saved["metrics"]["good"] == result.metrics["good"].model_dump()
    assert saved["validation_metrics"].keys() == result.validation_metrics.keys()


def test_training_run_is_recorded(service, runs, stub_candidates):
    service.train()

    run = runs.get("run-1")
    assert run["kind"] == "train"
    assert run["status"] == "ok"
    assert run["rows"] > 0


def test_tune_uses_the_tuned_candidates(service, stub_candidates, monkeypatch):
    received = []
    tuned = {"tuned": OffsetModel(0.0)}
    monkeypatch.setattr(training, "build_tuned_candidates", lambda split: received.append(split) or tuned)

    result = service.train(tune=True)

    assert len(received) == 1
    assert result.best_model == "tuned"


def test_training_without_observations_raises_and_records_error(store, runs, stub_candidates):
    service = TrainingService(InMemoryObservationRepository(), store, runs)

    with pytest.raises(InsufficientHistoryError, match="run `ingest` first"):
        service.train()

    assert runs.get("run-1")["status"] == "error"
    assert store.saves == 0


def test_training_without_test_data_raises(store, runs, stub_candidates, training_span_observations):
    # Observations only up to 2025: the test set (2026) is empty
    observations = validate_observations(training_span_observations)[0]
    repository = InMemoryObservationRepository()
    repository.upsert(observations[observations["time_utc"] <= VALIDATION_END])

    with pytest.raises(InsufficientHistoryError, match="test set is empty"):
        TrainingService(repository, store, runs).train()

    assert "InsufficientHistoryError" in runs.get("run-1")["error"]


def test_failing_model_marks_run_as_error(service, runs, store, monkeypatch):
    class BrokenModel(OffsetModel):
        def fit(self, X, y):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(training, "build_candidates", lambda: {"broken": BrokenModel()})

    with pytest.raises(RuntimeError):
        service.train()

    assert runs.get("run-1")["status"] == "error"
    assert store.saves == 0


def test_training_with_the_real_candidates(service, store, monkeypatch):
    # Real naive, Ridge and HGB (with few trees to be fast): the whole flow works end to end
    monkeypatch.setitem(modeling.HGB_PARAMS, "max_iter", 20)

    result = service.train()

    assert set(result.metrics) == {"naive", "ridge", "hgb"}
    assert result.best_model in result.metrics
    assert store.load().model is not None
