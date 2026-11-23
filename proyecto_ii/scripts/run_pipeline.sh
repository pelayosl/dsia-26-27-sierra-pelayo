#!/usr/bin/env bash
# Unattended run of the whole pipeline, meant for cron.
#
#   1. (optional) refresh the dataset: `build_dataset --update` in dsia-dataset, if DSIA_DATASET_DIR is set
#   2. python -m energy_forecast.cli.pipeline   (ingest → train when needed → forecast)
#
# Every step only runs if the previous one succeeded (set -e + exit codes), and everything goes to
# logs/pipeline_<date>.log. Extra arguments are passed on to the pipeline CLI, e.g. `--train always`.
#
# Environment variables:
#   PYTHON            interpreter with the requirements installed (default: python). Under cron, use the
#                     absolute path of the virtual environment, e.g. /home/me/proyecto_ii/.venv/bin/python
#   DSIA_DATASET_DIR  path of the dsia-dataset clone; if empty, the dataset is not refreshed
#   LOG_DIR           folder for the logs (default: logs, inside proyecto_ii)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%F).log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | run_pipeline | $*" | tee -a "$LOG_FILE"; }

if [[ -n "${DSIA_DATASET_DIR:-}" ]]; then
    log "Refreshing dsia-dataset in $DSIA_DATASET_DIR"
    (cd "$DSIA_DATASET_DIR" && uv run python -m src.build_dataset --update) >>"$LOG_FILE" 2>&1 \
        || { log "Dataset refresh failed: pipeline not run"; exit 1; }
fi

log "Running the pipeline"
if "$PYTHON" -m energy_forecast.cli.pipeline --log-file "$LOG_FILE" "$@"; then
    log "Pipeline OK"
else
    code=$?
    log "Pipeline failed with exit code $code"
    exit "$code"
fi
