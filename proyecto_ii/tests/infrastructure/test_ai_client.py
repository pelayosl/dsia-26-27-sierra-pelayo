"""Tests of the AI edge (no network)."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from energy_forecast.domain.models import ForecastSummary
from energy_forecast.infrastructure.ai_client import AIClient, MockClient, get_client

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
