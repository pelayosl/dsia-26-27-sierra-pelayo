"""Pipeline orchestrator: `ingest` → `train` (when needed) → `forecast`, stopping at the first failure.

Usage: python -m energy_forecast.cli.pipeline [--train {auto,always,never}] [--no-explain] [--log-file PATH] [-v]

It runs the same CLIs a person would run by hand, in order. A step only runs if the previous one returned 0,
and the pipeline returns the code of the step that failed. 
Every step keeps its own error handling and is recorded in `pipeline_runs`.

Training is the slow step and the saved model does not need to change every day, so by
default (`--train auto`) it only runs when there is no trained model yet.

`scripts/run_pipeline.sh` wraps this command for cron (and optionally refreshes dsia-dataset first).
"""

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

from energy_forecast.cli import EXIT_INPUT_ERROR, EXIT_OK, forecast, ingest, load_cli_settings, train
from energy_forecast.logging_config import get_logger, setup_logging

logger = get_logger("cli.pipeline")

TRAIN_AUTO, TRAIN_ALWAYS, TRAIN_NEVER = "auto", "always", "never"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reads the arguments; with `argv=None` uses the command line."""
    parser = argparse.ArgumentParser(description="Run the whole pipeline: ingest → train (when needed) → forecast.")
    parser.add_argument(
        "--train", choices=[TRAIN_AUTO, TRAIN_ALWAYS, TRAIN_NEVER], default=TRAIN_AUTO,
        help="auto (default): train only if there is no trained model; always: retrain; never: skip training.",
    )
    parser.add_argument(
        "--no-explain", dest="explain", action="store_false",
        help="Forecast without the AI explanation (no call to the AI provider).",
    )
    parser.add_argument("--log-file", type=Path, help="Also write the log of every step to this file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging.")
    return parser.parse_args(argv)


def _common_args(args: argparse.Namespace) -> list[str]:
    """Logging arguments passed on to every step, so all of them write to the same log."""
    common = ["--log-file", str(args.log_file)] if args.log_file else []
    return common + (["-v"] if args.verbose else [])


def plan_steps(args: argparse.Namespace, model_exists: bool) -> list[tuple[str, Callable[[list[str]], int], list[str]]]:
    """Steps to run, in order: `(name, main function, argv)`."""
    common = _common_args(args)
    steps = [("ingest", ingest.main, common)]
    if args.train == TRAIN_ALWAYS or (args.train == TRAIN_AUTO and not model_exists):
        steps.append(("train", train.main, common))
    steps.append(("forecast", forecast.main, common + ([] if args.explain else ["--no-explain"])))
    return steps


def main(argv: list[str] | None = None) -> int:
    """Runs the steps in order and returns 0, or the exit code of the first step that failed (1 or 2)."""
    args = parse_args(argv)

    settings = load_cli_settings()
    if settings is None:
        return EXIT_INPUT_ERROR
    setup_logging("DEBUG" if args.verbose else settings.log_level, log_file=args.log_file)

    steps = plan_steps(args, model_exists=settings.model_path.exists())
    
    logger.info("Pipeline started: %s", " -> ".join(name for name, _, _ in steps))
    started = time.perf_counter()
    for name, step_main, step_argv in steps:
        step_started = time.perf_counter()
        code = step_main(step_argv)
        # Each step reconfigures logging with the same options; restore it for the orchestrator's own messages
        setup_logging("DEBUG" if args.verbose else settings.log_level, log_file=args.log_file)
        if code != EXIT_OK:
            logger.error("Pipeline stopped: step '%s' failed with exit code %s", name, code)
            return code
        logger.info("Step '%s' finished in %.1f s", name, time.perf_counter() - step_started)

    logger.info("Pipeline finished in %.1f s", time.perf_counter() - started)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
