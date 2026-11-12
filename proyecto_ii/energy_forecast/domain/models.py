"""Domain contracts (Pydantic) that cross layers and are serialized to JSON."""

from datetime import datetime, timedelta
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from energy_forecast.domain.schema import HORIZON_HOURS


class DomainModel(BaseModel):
    """Common base: immutable, with no undeclared extra fields."""
    # `protected_namespaces=()` allows fields such as `model_version` without Pydantic warnings
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class HourlyPrediction(DomainModel):
    """Demand prediction for a target hour."""
    time_utc: datetime
    predicted_mw: float = Field(ge=0)


class Forecast(DomainModel):
    """24 h forecast from an origin. Same shape in `reports/forecast_<date>.json` and in `/forecast`."""
    origin: datetime
    predictions: tuple[HourlyPrediction, ...] = Field(min_length=HORIZON_HOURS, max_length=HORIZON_HOURS)
    model_version: str
    explanation: str | None = None
    provider: str | None = None
    ai_model: str | None = None

    @model_validator(mode="after")
    def _check_consecutive_utc_hours(self) -> Self:
        """UTC-aware datetimes and predictions in consecutive hours starting at `origin + 1 h`."""
        for label, value in (("origin", self.origin),
                             *((f"predictions[{i}].time_utc", p.time_utc) for i, p in enumerate(self.predictions))):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError(f"{label} debe tener zona horaria UTC")
        for i, prediction in enumerate(self.predictions, start=1):
            expected = self.origin + timedelta(hours=i)
            if prediction.time_utc != expected:
                raise ValueError(
                    f"predictions[{i - 1}].time_utc debe ser {expected.isoformat()}, "
                    f"es {prediction.time_utc.isoformat()}"
                )
        return self


class ForecastSummary(DomainModel):
    """Forecast data already computed by the domain; the LLM only writes them up."""
    peak_time_local: datetime
    peak_mw: float = Field(ge=0)
    valley_time_local: datetime
    valley_mw: float = Field(ge=0)
    mean_mw: float = Field(ge=0)
    change_vs_last_week_pct: float
    mean_temperature_c: float
    is_holiday: bool
    is_weekend: bool
    model_mae_mw: float = Field(ge=0)


class IngestionReport(DomainModel):
    """Result of an ingestion."""
    rows_in: int = Field(ge=0)
    rows_valid: int = Field(ge=0)
    rows_rejected: int = Field(ge=0)
    rows_written: int = Field(ge=0)


class ModelMetrics(DomainModel):
    """Metrics of a model on a set of hours (MAE and RMSE in MW, MAPE in %)."""
    mae: float = Field(ge=0)
    rmse: float = Field(ge=0)
    mape: float = Field(ge=0)


class TrainingResult(DomainModel):
    """Result of a training run: metrics per model and version of the persisted model.

    `best_model` is chosen by `validation_metrics` (fitted on train); `metrics` are the test metrics of each
    model fitted on train + validation, which is the persisted one. The test set is not used to choose.
    """
    model_version: str
    best_model: str
    validation_metrics: dict[str, ModelMetrics]
    metrics: dict[str, ModelMetrics]
    mae_by_hour: dict[int, float]
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    features: tuple[str, ...]
