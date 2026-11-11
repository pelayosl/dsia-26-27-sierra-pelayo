"""Persistence of the trained model and its metrics.

`model.joblib` holds everything needed to predict (model, version, metrics, sklearn version), so
`load` never mixes a model with the metrics of another one. `metrics.json` is a readable copy of the
metrics for people, the README and the dashboard.
"""

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import joblib
import sklearn
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python
from sklearn.base import BaseEstimator

from energy_forecast.domain.errors import ModelNotFoundError
from energy_forecast.logging_config import get_logger

logger = get_logger("model_store")

VERSION_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
PAYLOAD_KEYS = {"model", "version", "metrics", "sklearn_version"}


class StoredModel(BaseModel):
    """Loaded model together with its version and metrics."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, protected_namespaces=())

    model: BaseEstimator
    version: str
    metrics: dict[str, Any]


class ModelStore(Protocol):
    """Model store abstraction."""
    def save(self, model: BaseEstimator, name: str, metrics: dict[str, Any]) -> str:
        """Saves the model and its metrics; returns the assigned version (`<name>-<UTC timestamp>`)."""

    def load(self) -> StoredModel:
        """Loads the last saved model; raises `ModelNotFoundError` if there is none."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Writes to a temporary file and then replaces `path`: a failed save never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class JoblibModelStore:
    """Saves the `Pipeline` in `models/model.joblib` and the metrics in `reports/metrics.json`."""

    def __init__(self, model_path: Path, metrics_path: Path, clock: Callable[[], datetime] = _utc_now) -> None:
        self._model_path = Path(model_path)
        self._metrics_path = Path(metrics_path)
        self._clock = clock  # injectable so tests get predictable versions

    def save(self, model: BaseEstimator, name: str, metrics: dict[str, Any]) -> str:
        """`metrics` must be JSON-serialisable once converted (Pydantic models, datetimes and tuples are fine).

        From a `TrainingResult`, pass `result.model_dump(exclude={"model_version"})`.
        """
        saved_at = self._clock().astimezone(timezone.utc)
        version = f"{name}-{saved_at.strftime(VERSION_TIME_FORMAT)}"
        report = to_jsonable_python({"model_version": version, "saved_at": saved_at, **metrics})
        report_json = json.dumps(report, indent=2, ensure_ascii=False)

        # The same JSON types in the payload and in `metrics.json`: `load` returns exactly what the file shows
        payload = {"model": model, "version": version, "metrics": report, "sklearn_version": sklearn.__version__}
        _write_atomically(self._model_path, lambda path: joblib.dump(payload, path))
        _write_atomically(self._metrics_path, lambda path: path.write_text(report_json + "\n", encoding="utf-8"))
        logger.info("Model %s saved in %s", version, self._model_path)
        return version

    def load(self) -> StoredModel:
        if not self._model_path.exists():
            raise ModelNotFoundError(f"No trained model in {self._model_path}: run `train` first")
        try:
            payload = joblib.load(self._model_path)
        except Exception as exc:  # pickle can fail in many ways (truncated file, missing class, ...)
            raise ModelNotFoundError(
                f"{self._model_path} could not be loaded ({type(exc).__name__}: {exc}): run `train` again"
            ) from exc
        if not isinstance(payload, dict) or not PAYLOAD_KEYS <= payload.keys():
            raise ModelNotFoundError(f"{self._model_path} is not a model saved by `train`: run `train` again")

        if payload["sklearn_version"] != sklearn.__version__:
            logger.warning(
                "Model %s was trained with scikit-learn %s and is loaded with %s: retrain if predictions look wrong",
                payload["version"], payload["sklearn_version"], sklearn.__version__,
            )
        return StoredModel(model=payload["model"], version=payload["version"], metrics=payload["metrics"])
