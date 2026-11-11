"""Use case: load the dsia-dataset CSV, validate it and store it in the database."""

from energy_forecast.domain.models import IngestionReport
from energy_forecast.domain.validator import REJECTION_REASON_COLUMN, validate_observations
from energy_forecast.infrastructure.dataset_source import DatasetSource
from energy_forecast.infrastructure.repositories import ObservationRepository, PipelineRunRepository
from energy_forecast.logging_config import get_logger

logger = get_logger("ingestion")


class IngestionService:
    """Orchestrates `DatasetSource.read` → `validate_observations` → `ObservationRepository.upsert`."""

    def __init__(
        self,
        source: DatasetSource,
        observations: ObservationRepository,
        runs: PipelineRunRepository,
    ) -> None:
        self._source = source
        self._observations = observations
        self._runs = runs

    def run(self) -> IngestionReport:
        """Runs the whole ingestion and returns its report.

        The run is recorded in `pipeline_runs`. On any error the run is marked as `error` and the exception
        is re-raised, so the caller (CLI or API) decides the exit code / HTTP status.
        """
        run_id = self._runs.start("ingest")
        logger.info("Ingestion started (run %s)", run_id)
        try:
            raw = self._source.read()
            observations, rejected = validate_observations(raw)
            written = self._observations.upsert(observations)
        except Exception as exc:
            self._runs.finish(run_id, "error", error=f"{type(exc).__name__}: {exc}")
            logger.error("Ingestion failed (run %s): %s", run_id, exc)
            raise

        self._runs.finish(run_id, "ok", rows=written)
        report = IngestionReport(
            rows_in=len(raw),
            rows_valid=len(observations),
            rows_rejected=len(rejected),
            rows_written=written,
        )
        logger.info(
            "Ingestion finished (run %s): read=%s valid=%s rejected=%s written=%s",
            run_id, report.rows_in, report.rows_valid, report.rows_rejected, report.rows_written,
        )
        if not rejected.empty:
            reasons = rejected[REJECTION_REASON_COLUMN].value_counts().to_dict()
            logger.warning("Rejected rows by reason: %s", reasons)
        return report
