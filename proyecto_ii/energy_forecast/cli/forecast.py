"""Forecast CLI: sklearn inference (+ AI explanation from step 4.4) → `reports/forecast_<date>.json`.

Usage: python -m energy_forecast.cli.forecast [--origin 2026-11-07T23:00:00+00:00] [--log-file PATH] [-v]
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from energy_forecast.bootstrap import build_forecast_service
from energy_forecast.cli import EXIT_FAILURE, EXIT_INPUT_ERROR, EXIT_OK, load_cli_settings
from energy_forecast.domain.errors import ExternalServiceError, InsufficientHistoryError, ModelNotFoundError
from energy_forecast.domain.models import Forecast
from energy_forecast.domain.schema import LOCAL_TIMEZONE
from energy_forecast.logging_config import get_logger, setup_logging

logger = get_logger("cli.forecast")

REPORT_TIME_FORMAT = "%Y%m%dT%H"


def parse_origin(value: str) -> datetime:
    """ISO 8601 origin for `--origin`, on the hour; without a timezone it is taken as UTC (as in the database).

    An invalid value makes argparse exit with code 2, the same as `EXIT_INPUT_ERROR`.
    """
    try:
        origin = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO 8601 datetime: '{value}'") from exc
    if (origin.minute, origin.second, origin.microsecond) != (0, 0, 0):
        raise argparse.ArgumentTypeError(f"the origin must be on the hour: '{value}'")
    return origin.replace(tzinfo=timezone.utc) if origin.tzinfo is None else origin


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reads the arguments; with `argv=None` uses the command line."""
    parser = argparse.ArgumentParser(description="Forecast the electricity demand of the next 24 hours.")
    parser.add_argument(
        "--origin", type=parse_origin,
        help="Last known hour, ISO 8601 (default: the last hour with demand in the database). Without timezone: UTC.",
    )
    parser.add_argument("--log-file", type=Path, help="Also write the log to this file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return parser.parse_args(argv)


def report_path(reports_dir: Path, forecast: Forecast) -> Path:
    """`reports/forecast_<origin in UTC, YYYYMMDDTHH>.json`: one report per origin (a rerun overwrites it)."""
    return reports_dir / f"forecast_{forecast.origin.astimezone(timezone.utc).strftime(REPORT_TIME_FORMAT)}.json"


def write_report(forecast: Forecast, path: Path) -> None:
    """Contract 2 (§7 of ARCHITECTURE.md): the same JSON shape as the `/forecast` response of Project III."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(forecast.model_dump_json(indent=2) + "\n", encoding="utf-8")


def format_forecast(forecast: Forecast) -> str:
    """The 24 predictions in local time, for the terminal."""
    lines = [f"Forecast from {forecast.origin.isoformat()} (model {forecast.model_version}), local time:"]
    for prediction in forecast.predictions:
        local = prediction.time_utc.astimezone(ZoneInfo(LOCAL_TIMEZONE))
        lines.append(f"  {local:%Y-%m-%d %H:%M}  {prediction.predicted_mw:>9,.0f} MW")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Runs the forecast, writes the JSON report and returns the exit code.

    0 ok · 2 no trained model, not enough history or invalid `--origin` · 1 weather service down or unexpected error.
    """
    args = parse_args(argv)

    settings = load_cli_settings()
    if settings is None:
        return EXIT_INPUT_ERROR
    setup_logging("DEBUG" if args.verbose else settings.log_level, log_file=args.log_file)

    try:
        forecast = build_forecast_service(settings).forecast(origin=args.origin)
    except (ModelNotFoundError, InsufficientHistoryError) as exc:
        logger.error("Cannot forecast: %s", exc)
        return EXIT_INPUT_ERROR
    except ExternalServiceError as exc:
        logger.error("External service unavailable: %s", exc)
        return EXIT_FAILURE
    except Exception:
        logger.exception("Unexpected error during the forecast")
        return EXIT_FAILURE

    path = report_path(settings.reports_dir, forecast)
    write_report(forecast, path)
    print(format_forecast(forecast))
    print(f"\nReport written to {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
