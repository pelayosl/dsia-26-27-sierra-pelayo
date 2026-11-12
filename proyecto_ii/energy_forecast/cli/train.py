"""Training CLI: database → compared models → `models/model.joblib` + `reports/metrics.json`.

Usage: python -m energy_forecast.cli.train [--tune] [--log-file PATH] [-v]
"""

import argparse
import sys
from pathlib import Path

from energy_forecast.bootstrap import build_training_service
from energy_forecast.cli import EXIT_FAILURE, EXIT_INPUT_ERROR, EXIT_OK, load_cli_settings
from energy_forecast.domain.errors import InsufficientHistoryError
from energy_forecast.domain.models import TrainingResult
from energy_forecast.logging_config import get_logger, setup_logging

logger = get_logger("cli.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reads the arguments; with `argv=None` uses the command line."""
    parser = argparse.ArgumentParser(description="Train and compare the models and save the best one.")
    parser.add_argument(
        "--tune", action="store_true",
        help="Tune the Ridge and HGB hyperparameters with TimeSeriesSplit first (slower).",
    )
    parser.add_argument("--log-file", type=Path, help="Also write the log to this file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return parser.parse_args(argv)


def format_metrics_table(result: TrainingResult) -> str:
    """Readable table with the validation and test metrics of every model; the saved one is marked."""
    lines = [
        f"{'model':<8}{'val MAE':>10}{'test MAE':>10}{'test RMSE':>11}{'test MAPE':>11}",
        "-" * 50,
    ]
    for name, test in result.metrics.items():
        validation = result.validation_metrics[name]
        marker = "  <- saved" if name == result.best_model else ""
        lines.append(f"{name:<8}{validation.mae:>10.0f}{test.mae:>10.0f}{test.rmse:>11.0f}{test.mape:>10.2f}%{marker}")
    lines.append(f"\nModel version: {result.model_version} (MAE and RMSE in MW)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Runs the training, prints the metrics table and returns the exit code (0 ok, 1 failure, 2 no data)."""
    args = parse_args(argv)

    settings = load_cli_settings()
    if settings is None:
        return EXIT_INPUT_ERROR
    setup_logging("DEBUG" if args.verbose else settings.log_level, log_file=args.log_file)

    try:
        result = build_training_service(settings).train(tune=args.tune)
    except InsufficientHistoryError as exc:
        logger.error("Not enough data to train: %s", exc)
        return EXIT_INPUT_ERROR
    except Exception:
        logger.exception("Unexpected error during training")
        return EXIT_FAILURE

    print(format_metrics_table(result))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
