"""AI edge: natural-language explanation of the forecast (mock by default; NVIDIA API Catalog optional, D13).

The explanation is written in English. The figures come already computed
in the `ForecastSummary`: the clients only write them up.
"""

from datetime import datetime
from typing import Protocol

from energy_forecast.config import Settings
from energy_forecast.domain.models import ForecastSummary

# Used by the real client
SYSTEM_PROMPT = (
    "You are a power-system analyst. Explain the demand forecast in English, in 3-4 sentences. "
    "Use only the data provided and do not invent figures."
)
MAX_TOKENS = 200
TEMPERATURE = 0.3
REQUEST_TIMEOUT_S = 20


class AIClient(Protocol):
    """Minimal interface of the AI edge."""
    name: str
    model: str | None

    def explain(self, summary: ForecastSummary) -> str:
        """Returns a short explanation of the forecast from its numeric summary."""


def _mw(value: float) -> str:
    """30000 → '30,000 MW'."""
    return f"{value:,.0f} MW"


def _when(moment: datetime) -> str:
    """Local datetime → 'on Jan 08 at 20:00'."""
    return f"on {moment:%b %d} at {moment:%H:%M}"


def _number(value: float, decimals: int = 1) -> str:
    """12.5 → '12.5'."""
    return f"{value:.{decimals}f}"


class MockClient:
    """Deterministic client without network: fills an English template with the summary. Default in tests and CI."""
    name = "mock"
    model = None

    def explain(self, summary: ForecastSummary) -> str:
        """The same summary always produces the same text; every figure comes from the summary."""
        sentences = [
            f"From {_when(summary.period_start_local)} to {_when(summary.period_end_local)}, the average demand "
            f"is forecast at {_mw(summary.mean_mw)}.",
            f"The peak will be {_when(summary.peak_time_local)} ({_mw(summary.peak_mw)}) and the valley "
            f"{_when(summary.valley_time_local)} ({_mw(summary.valley_mw)}).",
        ]

        if summary.change_vs_last_week_pct is None:
            sentences.append("There is no data from the previous week to compare against.")
        elif summary.change_vs_last_week_pct == 0:
            sentences.append("This is the same demand as in the same hours of the previous week.")
        else:
            direction = "higher" if summary.change_vs_last_week_pct > 0 else "lower"
            sentences.append(f"This is {_number(abs(summary.change_vs_last_week_pct))}% {direction} than in the "
                             f"same hours of the previous week.")
        sentences.append(f"The forecast mean temperature is {_number(summary.mean_temperature_c)} °C.")

        context = [label for flag, label in ((summary.is_holiday, "a national holiday"),
                                             (summary.is_weekend, "a weekend")) if flag]
        if context:
            sentences.append(f"The period includes {' and '.join(context)}, which usually lowers demand.")

        sentences.append(f"Mean model error on the test set: ±{_mw(summary.model_mae_mw)}.")
        return " ".join(sentences)


def get_client(settings: Settings) -> AIClient:
    """Returns the client for `AI_PROVIDER`: `mock` (by default) or the real provider."""
    if settings.ai_provider == "mock":
        return MockClient()
    # TODO (step 4.3): "nvidia" → NvidiaClient(settings.nvidia_api_key, settings.nvidia_model) (D13).
    raise ValueError(f"Unsupported AI provider '{settings.ai_provider}': use 'mock'")
