"""Pipeline CLIs (Project II interface): `ingest`, `train` and `forecast`."""

# Exit codes (same convention as the session 5 job)
EXIT_OK = 0
EXIT_FAILURE = 1        # system or external service failure
EXIT_INPUT_ERROR = 2    # bad input: missing file, invalid schema, no trained model
