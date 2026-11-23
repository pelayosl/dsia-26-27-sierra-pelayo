"""AI edge: natural-language explanation of the forecast (mock by default; NVIDIA API Catalog optional, D13).

The explanation is written in English. The figures come already computed
in the `ForecastSummary`: the clients only write them up.
"""

import re
from datetime import datetime
from typing import Protocol

from huggingface_hub import InferenceClient

from energy_forecast.config import Settings
from energy_forecast.domain.errors import ExternalServiceError
from energy_forecast.domain.models import ForecastSummary

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Reasoning that some models write inside the answer; removed just in case reasoning gets enabled
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

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
    return f"{moment:%b %d} at {moment:%H:%M}"


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
            f"The peak will be on {_when(summary.peak_time_local)} ({_mw(summary.peak_mw)}) and the valley "
            f"on {_when(summary.valley_time_local)} ({_mw(summary.valley_mw)}).",
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

        if summary.model_mae_mw is not None:
            sentences.append(f"Mean model error on the test set: ±{_mw(summary.model_mae_mw)}.")
        return " ".join(sentences)


class NvidiaClient:
    """NVIDIA API Catalog client (D13): `chat_completion` through `huggingface_hub.InferenceClient`.

    The endpoint is OpenAI-compatible, so the course's `InferenceClient` works with `base_url`.
    Nemotron is a reasoning model: reasoning is disabled (`enable_thinking=False`). With it enabled, a manual
    test spent 196 of the 200 tokens reasoning and returned a cut-off answer.
    Any provider failure (network, timeout, 4xx/5xx, empty or cut-off answer) raises `ExternalServiceError`,
    so the forecast service can fall back to the mock.
    """
    name = "nvidia"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = REQUEST_TIMEOUT_S,
        inference_client: InferenceClient | None = None,
    ) -> None:
        self.model = model
        # Injectable so tests can replace the HTTP client with a fake
        self._client = inference_client or InferenceClient(base_url=NVIDIA_BASE_URL, api_key=api_key, timeout=timeout_s)

    def explain(self, summary: ForecastSummary) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": summary.model_dump_json()},
        ]
        try:
            output = self._client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as exc:  # network errors, timeouts and HTTP errors of `huggingface_hub`
            # HTTP errors carry the response: its status (401 invalid key, 429 rate limit…) says more than the text
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f"HTTP {status}, " if status else ""
            raise ExternalServiceError(f"NVIDIA API request failed ({detail}{type(exc).__name__}): {exc}") from exc

        if not output.choices:
            raise ExternalServiceError("NVIDIA API returned no choices")
        choice = output.choices[0]
        text = _THINK_BLOCK.sub("", choice.message.content or "").strip()
        if not text:
            raise ExternalServiceError("NVIDIA API returned an empty explanation")
        if choice.finish_reason == "length":
            raise ExternalServiceError(f"NVIDIA API explanation was cut off at {MAX_TOKENS} tokens")
        return text


def get_client(settings: Settings) -> AIClient:
    """Returns the client for `AI_PROVIDER`: `mock` (by default) or `nvidia` (D13)."""
    if settings.ai_provider == "mock":
        return MockClient()
    if settings.ai_provider == "nvidia":
        return NvidiaClient(settings.nvidia_api_key.get_secret_value(), settings.nvidia_model)
    raise ValueError(f"Unsupported AI provider '{settings.ai_provider}': use 'mock' or 'nvidia'")
