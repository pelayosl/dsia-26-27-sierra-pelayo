"""Use case: train and compare the candidate models and persist the best one."""

from sklearn.base import BaseEstimator, clone

from energy_forecast.domain.errors import InsufficientHistoryError
from energy_forecast.domain.evaluation import evaluate, mae_by_hour
from energy_forecast.domain.features import FEATURE_COLUMNS, build_training_frame
from energy_forecast.domain.models import ModelMetrics, TrainingResult
from energy_forecast.domain.modeling import build_candidates, build_tuned_candidates, temporal_split
from energy_forecast.infrastructure.model_store import ModelStore
from energy_forecast.infrastructure.repositories import ObservationRepository, PipelineRunRepository
from energy_forecast.logging_config import get_logger

logger = get_logger("training")


class TrainingService:
    """Orchestrates: observations → features → temporal split → training → evaluation → `ModelStore`."""

    def __init__(
        self,
        observations: ObservationRepository,
        model_store: ModelStore,
        runs: PipelineRunRepository,
    ) -> None:
        self._observations = observations
        self._model_store = model_store
        self._runs = runs

    def train(self, tune: bool = False) -> TrainingResult:
        """Trains every candidate, chooses the best one by validation MAE and saves it.

        For each candidate:
        1. fit on train → metrics on validation (used to choose the best model),
        2. fit on train + validation → metrics on test (reported; the test set never takes part in choosing).
        The persisted model is the best candidate fitted on train + validation, so `metrics.json` describes
        exactly the saved model.

        The run is recorded in `pipeline_runs`. On any error it is marked as `error` and the exception is
        re-raised. `InsufficientHistoryError` means the database does not cover train, validation and test.
        """
        run_id = self._runs.start("train")
        logger.info("Training started (run %s, tune=%s)", run_id, tune)
        try:
            result, rows = self._train(tune)
        except Exception as exc:
            self._runs.finish(run_id, "error", error=f"{type(exc).__name__}: {exc}")
            logger.error("Training failed (run %s): %s", run_id, exc)
            raise

        self._runs.finish(run_id, "ok", rows=rows)
        logger.info("Training finished (run %s): best model %s saved as %s",
                    run_id, result.best_model, result.model_version)
        return result

    def _train(self, tune: bool) -> tuple[TrainingResult, int]:
        observations = self._observations.range()
        if observations.empty:
            raise InsufficientHistoryError("There are no observations in the database: run `ingest` first")
        X, y = build_training_frame(observations)
        try:
            split = temporal_split(X, y)
        except ValueError as exc:
            raise InsufficientHistoryError(str(exc)) from exc
        logger.info("Rows: train=%s validation=%s test=%s", len(split.X_train), len(split.X_val), len(split.X_test))

        candidates = build_tuned_candidates(split) if tune else build_candidates()
        validation_metrics: dict[str, ModelMetrics] = {}
        test_metrics: dict[str, ModelMetrics] = {}
        fitted: dict[str, BaseEstimator] = {}
        for name, candidate in candidates.items():
            on_train = clone(candidate).fit(split.X_train, split.y_train)
            validation_metrics[name] = evaluate(split.y_val, on_train.predict(split.X_val))

            fitted[name] = clone(candidate).fit(split.X_train_val, split.y_train_val)
            test_metrics[name] = evaluate(split.y_test, fitted[name].predict(split.X_test))
            logger.info("%s: validation MAE %.0f MW | test MAE %.0f MW, RMSE %.0f MW, MAPE %.2f %%",
                        name, validation_metrics[name].mae, test_metrics[name].mae,
                        test_metrics[name].rmse, test_metrics[name].mape)

        best = min(validation_metrics, key=lambda name: validation_metrics[name].mae)
        best_model = fitted[best]
        details = {
            "best_model": best,
            "validation_metrics": validation_metrics,
            "metrics": test_metrics,
            "mae_by_hour": mae_by_hour(split.y_test, best_model.predict(split.X_test)),
            "train_start": split.X_train.index.min().to_pydatetime(),
            "train_end": split.X_val.index.max().to_pydatetime(),
            "test_start": split.X_test.index.min().to_pydatetime(),
            "test_end": split.X_test.index.max().to_pydatetime(),
            "features": tuple(FEATURE_COLUMNS),
        }
        version = self._model_store.save(best_model, best, details)
        return TrainingResult(model_version=version, **details), len(split.X_train_val)
