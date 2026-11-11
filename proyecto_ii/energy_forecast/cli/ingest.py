"""Ingestion CLI: dsia-dataset CSV → database.

Usage: python -m energy_forecast.cli.ingest [--input PATH_OR_URL] [--log-file PATH] [-v]
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError as ConfigError

from energy_forecast.bootstrap import build_ingestion_service
from energy_forecast.cli import EXIT_FAILURE, EXIT_INPUT_ERROR, EXIT_OK
from energy_forecast.config import load_settings
from energy_forecast.domain.errors import DataLoadError, ValidationError
from energy_forecast.logging_config import get_logger, setup_logging

logger = get_logger("cli.ingest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reads the arguments; with `argv=None` uses the command line."""
    parser = argparse.ArgumentParser(description="Load the dsia-dataset CSV, validate it and store it in the database.")
    parser.add_argument("--input", help="Path or URL of the CSV (default: DATASET_PATH from the configuration).")
    parser.add_argument("--log-file", type=Path, help="Also write the log to this file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs the ingestion and returns the exit code (0 ok, 1 system failure, 2 broken input)."""
    args = parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if args.input:
        settings = settings.model_copy(update={"dataset_path": args.input})

    setup_logging("DEBUG" if args.verbose else settings.log_level, log_file=args.log_file)

    try:
        build_ingestion_service(settings).run()
    except (DataLoadError, ValidationError) as exc:
        logger.error("Invalid input: %s", exc)
        return EXIT_INPUT_ERROR
    except Exception:
        logger.exception("Unexpected error during ingestion")
        return EXIT_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
