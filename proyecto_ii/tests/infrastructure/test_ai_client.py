"""Tests of the AI edge. No network, except `test_nvidia_real_request` (marked `integration`)."""

import json
import os
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from dotenv import dotenv_values

from energy_forecast.config import PROJECT_DIR, Settings
from energy_forecast.domain.errors import ExternalServiceError
from energy_forecast.domain.models import ForecastSummary
from energy_forecast.infrastructure.ai_client import (
    MAX_TOKENS,
    SYSTEM_PROMPT,
    AIClient,
    MockClient,
    NvidiaClient,
    get_client,
)

MADRID = ZoneInfo("Europe/Madrid")


def make_summary(**changes) -> ForecastSummary:
    values = {
        "period_start_local": datetime(2025, 1, 8, 1, 0, tzinfo=MADRID),
        "period_end_local": datetime(2025, 1, 9, 0, 0, tzinfo=MADRID),
        "peak_time_local": datetime(2025, 1, 8, 20, 0, tzinfo=MADRID),
        "peak_mw": 34_512,
        "valley_time_local": datetime(2025, 1, 8, 4, 0, tzinfo=MADRID),
        "valley_mw": 22_105,
        "mean_mw": 28_730,
        "change_vs_last_week_pct": 3.4,
        "mean_temperature_c": 8.6,
        "is_holiday": False,
        "is_weekend": False,
        "model_mae_mw": 640,
    }
    return ForecastSummary(**{**values, **changes})


# ---------------------------------------------------------------- MockClient


def test_mock_client_is_deterministic():
    summary = make_summary()

    assert MockClient().explain(summary) == MockClient().explain(summary)


def test_mock_explanation_contains_the_summary_figures():
    text = MockClient().explain(make_summary())

    assert "34,512 MW" in text and "on Jan 08 at 20:00" in text  # peak
    assert "22,105 MW" in text and "on Jan 08 at 04:00" in text  # valley
    assert "28,730 MW" in text  # mean
    assert "3.4% higher" in text
    assert "8.6 °C" in text
    assert "±640 MW" in text


def test_mock_explanation_with_lower_demand_than_last_week():
    assert "2.5% lower" in MockClient().explain(make_summary(change_vs_last_week_pct=-2.5))


def test_mock_explanation_without_last_week_data():
    text = MockClient().explain(make_summary(change_vs_last_week_pct=None))

    assert "There is no data from the previous week" in text
    assert "None" not in text


def test_mock_explanation_with_no_change():
    assert "the same demand" in MockClient().explain(make_summary(change_vs_last_week_pct=0.0))


@pytest.mark.parametrize(
    ("is_holiday", "is_weekend", "expected"),
    [
        (True, False, "includes a national holiday,"),
        (False, True, "includes a weekend,"),
        (True, True, "includes a national holiday and a weekend,"),
    ],
)
def test_mock_explanation_mentions_holiday_and_weekend(is_holiday, is_weekend, expected):
    text = MockClient().explain(make_summary(is_holiday=is_holiday, is_weekend=is_weekend))

    assert expected in text


def test_mock_explanation_on_an_ordinary_day_does_not_mention_the_calendar():
    text = MockClient().explain(make_summary())

    assert "holiday" not in text and "weekend" not in text


def test_mock_client_identifies_itself():
    client = MockClient()

    assert client.name == "mock"
    assert client.model is None


# ---------------------------------------------------------------- get_client


def test_get_client_defaults_to_mock(mock_settings):
    client: AIClient = get_client(mock_settings)

    assert isinstance(client, MockClient)
    assert client.name == "mock"


def test_get_client_unsupported_provider_raises(mock_settings):
    # `Settings` only accepts known providers, so an unsupported one is simulated with `model_copy`
    settings = mock_settings.model_copy(update={"ai_provider": "other"})

    with pytest.raises(ValueError, match="Unsupported AI provider 'other'"):
        get_client(settings)


def test_get_client_nvidia():
    settings = Settings(_env_file=None, ai_provider="nvidia", nvidia_api_key="nvapi-test", nvidia_model="owner/model")

    client = get_client(settings)

    assert isinstance(client, NvidiaClient)
    assert (client.name, client.model) == ("nvidia", "owner/model")


# ---------------------------------------------------------------- NvidiaClient (fake InferenceClient)


class FakeInferenceClient:
    """Records the `chat_completion` calls and returns `answer`, or raises `error`."""

    def __init__(self, content: str | None = "The demand peaks at 20:00.", finish_reason: str = "stop",
                 error: Exception | None = None, choices: bool = True) -> None:
        self._content, self._finish_reason, self._error, self._choices = content, finish_reason, error, choices
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        choices = [SimpleNamespace(message=message, finish_reason=self._finish_reason)] if self._choices else []
        return SimpleNamespace(choices=choices)


def nvidia_client(**fake_options) -> tuple[NvidiaClient, FakeInferenceClient]:
    fake = FakeInferenceClient(**fake_options)
    return NvidiaClient("nvapi-test", "owner/model", inference_client=fake), fake


def test_nvidia_client_returns_the_answer():
    client, _ = nvidia_client(content="  The demand peaks at 20:00.  ")

    assert client.explain(make_summary()) == "The demand peaks at 20:00."


def test_nvidia_request_sends_prompt_summary_and_disables_reasoning():
    client, fake = nvidia_client()
    summary = make_summary()

    client.explain(summary)

    call = fake.calls[0]
    assert call["model"] == "owner/model"
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert json.loads(call["messages"][1]["content"]) == summary.model_dump(mode="json")
    assert call["max_tokens"] == MAX_TOKENS
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_nvidia_reasoning_block_is_removed():
    client, _ = nvidia_client(content="<think>let me compute...</think>The peak is at 20:00.")

    assert client.explain(make_summary()) == "The peak is at 20:00."


@pytest.mark.parametrize(
    ("fake_options", "message"),
    [
        ({"error": TimeoutError("read timed out")}, "request failed"),
        ({"error": RuntimeError("429 Too Many Requests")}, "429"),
        ({"content": None}, "empty explanation"),
        ({"content": "   "}, "empty explanation"),
        ({"content": "The system peaked", "finish_reason": "length"}, "cut off"),
        ({"choices": False}, "no choices"),
    ],
    ids=["timeout", "http-error", "none", "blank", "cut-off", "no-choices"],
)
def test_nvidia_failures_raise_external_service_error(fake_options, message):
    client, _ = nvidia_client(**fake_options)

    with pytest.raises(ExternalServiceError, match=message):
        client.explain(make_summary())


# ---------------------------------------------------------------- real API


def _real_api_key() -> str | None:
    return os.environ.get("NVIDIA_API_KEY") or dotenv_values(PROJECT_DIR / ".env").get("NVIDIA_API_KEY") or None


@pytest.mark.integration
@pytest.mark.skipif(_real_api_key() is None, reason="NVIDIA_API_KEY not configured")
def test_nvidia_real_request():
    client = NvidiaClient(_real_api_key(), Settings(_env_file=None).nvidia_model)

    text = client.explain(make_summary())

    assert len(text) > 20
    assert "<think>" not in text


def test_mock_explanation_without_model_error():
    text = MockClient().explain(make_summary(model_mae_mw=None))

    assert "model error" not in text and "None" not in text


def test_nvidia_http_error_includes_the_status_code():
    class HTTPError(Exception):
        response = SimpleNamespace(status_code=401)

    client, _ = nvidia_client(error=HTTPError("Request ID: abc"))

    with pytest.raises(ExternalServiceError, match="HTTP 401"):
        client.explain(make_summary())
