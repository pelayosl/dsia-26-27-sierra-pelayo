"""Common logging configuration for CLIs and API."""

import logging
from pathlib import Path

# Example: 026-11-10 09:30:01,123 | INFO | energy_forecast.train | Model trained
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Sets up logging configuration for the application.
    Args:
        level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file (Path | None): Optional path to a log file. If provided, logs will be written
        to this file in addition to the console.
    """ 
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    # force=True removes any existing logging configuration and applies the new one.
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)



def get_logger(name: str) -> logging.Logger:
    """Returns a named logger under the `energy_forecast` namespace."""
    return logging.getLogger(f"energy_forecast.{name}")
