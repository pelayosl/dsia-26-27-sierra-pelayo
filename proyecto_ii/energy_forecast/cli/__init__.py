"""Pipeline CLIs (Project II interface): `ingest`, `train` and `forecast`."""

import sys

from pydantic import ValidationError as ConfigError

from energy_forecast.config import Settings, load_settings

# Exit codes (same convention as the session 5 job)
EXIT_OK = 0
EXIT_FAILURE = 1        # system or external service failure
EXIT_INPUT_ERROR = 2    # bad input: missing file, invalid schema, no trained model


def load_cli_settings() -> Settings | None:
    """`load_settings()`, or `None` after printing the error if the configuration is invalid (→ exit 2).

    It is printed to stderr because logging is not configured yet: its level comes from the settings.
    """
    try:
        return load_settings()
    except ConfigError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return None
