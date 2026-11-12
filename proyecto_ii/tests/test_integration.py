"""Integration test: the whole pipeline (`ingest` → `train` → `forecast`) on the real dataset sample.

It runs the real CLIs, SQLite database, `JoblibModelStore` and sklearn models over the one-year sample of the
Project I (April 2025 - March 2026), which is reused instead of copying it to `proyecto_ii/data/`: it is the same
file that `test_validator.py` reads, and the full dataset is not in the repository (see the README).

Only the two network edges are replaced: Open-Meteo by `FakeWeatherForecastClient` and the AI provider by the
mock (`AI_PROVIDER=mock`). So it needs no network or keys, but it is marked `integration` because it is slower
(real HGB with 500 trees) and depends on a file outside `proyecto_ii/`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from energy_forecast import bootstrap
from energy_forecast.cli import EXIT_OK, forecast, ingest, train
from energy_forecast.domain import modeling
from energy_forecast.domain.schema import HORIZON_HOURS, LOAD_MW_RANGE
from tests.fakes import FakeWeatherForecastClient

SAMPLE_PATH = Path(__file__).parents[2] / "proyecto_i" / "data" / "energy_weather_es_hourly_sample.csv"

# The fixed split (train 2019-2024) does not fit a one-year sample: moved to train Apr-Oct 2025,
# validation Nov-Dec 2025 and test Jan-Mar 2026, keeping the same order and the same "never random" rule.
SAMPLE_TRAIN_END = pd.Timestamp("2025-10-31T23:00:00+00:00")
SAMPLE_VALIDATION_END = pd.Timestamp("2025-12-31T23:00:00+00:00")
# Last hour of the sample: the default origin of `forecast` (last hour with known demand)
SAMPLE_LAST_HOUR = datetime(2026, 3, 31, 21, tzinfo=timezone.utc)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("restore_root_logger"),
    pytest.mark.skipif(not SAMPLE_PATH.exists(), reason=f"Real sample not found: {SAMPLE_PATH}"),
]


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch, weather_forecast_24h):
    """Configuration in `tmp_path` reading the real sample, sample-sized split and Open-Meteo replaced by a fake."""
    monkeypatch.setenv("DATASET_PATH", str(SAMPLE_PATH))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'energy.db').as_posix()}")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "models" / "model.joblib"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AI_PROVIDER", "mock")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setattr(modeling, "TRAIN_END", SAMPLE_TRAIN_END)
    monkeypatch.setattr(modeling, "VALIDATION_END", SAMPLE_VALIDATION_END)
    weather = FakeWeatherForecastClient(weather_forecast_24h)
    monkeypatch.setattr(bootstrap, "OpenMeteoForecastClient", lambda: weather)
    return tmp_path, weather


def test_full_pipeline_on_real_sample(pipeline_env):
    tmp_path, weather = pipeline_env

    # Act: the three CLIs in order, as in the README
    assert ingest.main([]) == EXIT_OK
    assert train.main([]) == EXIT_OK
    assert forecast.main([]) == EXIT_OK

    # Training: model and metrics saved, and the saved model is the one with the best validation MAE.
    # Which one wins is not asserted: with only 7 months of training (no winter), the naive baseline can win
    # on this sample. The model quality is measured on the full dataset (README, `reports/metrics.json`).
    assert (tmp_path / "models" / "model.joblib").exists()
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["metrics"]) == {"naive", "ridge", "hgb"}
    validation = metrics["validation_metrics"]
    assert metrics["best_model"] == min(validation, key=lambda name: validation[name]["mae"])
    assert metrics["model_version"].startswith(f"{metrics['best_model']}-")
    assert (metrics["train_end"], metrics["test_start"]) == ("2025-12-31T23:00:00Z", "2026-01-01T00:00:00Z")

    # Forecast: contract 2 with 24 plausible hourly predictions and the mock explanation
    assert weather.origins == [SAMPLE_LAST_HOUR]
    report = json.loads((tmp_path / "reports" / "forecast_20260331T21.json").read_text(encoding="utf-8"))
    predictions = report["predictions"]
    assert len(predictions) == HORIZON_HOURS
    assert predictions[0]["time_utc"] == "2026-03-31T22:00:00Z"
    low, high = LOAD_MW_RANGE
    assert all(low <= prediction["predicted_mw"] <= high for prediction in predictions)
    assert report["model_version"] == metrics["model_version"]
    assert (report["provider"], report["ai_model"]) == ("mock", None)
    assert report["explanation"]
