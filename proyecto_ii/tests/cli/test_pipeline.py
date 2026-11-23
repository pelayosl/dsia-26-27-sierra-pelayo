"""Tests of the pipeline orchestrator (`cli/pipeline.py`): order of the steps, gate on exit codes, training policy."""

import json

import pytest

from energy_forecast.cli import EXIT_FAILURE, EXIT_INPUT_ERROR, EXIT_OK, forecast, ingest, pipeline, train

pytestmark = pytest.mark.usefixtures("restore_root_logger")


@pytest.fixture
def recorded_steps(monkeypatch):
    """Replaces the three CLIs with recorders that return 0; `codes` changes the code a step returns."""
    calls: list[tuple[str, list[str]]] = []
    codes = {"ingest": EXIT_OK, "train": EXIT_OK, "forecast": EXIT_OK}

    def recorder(name):
        def fake_main(argv):
            calls.append((name, argv))
            return codes[name]
        return fake_main

    for name, module in (("ingest", ingest), ("train", train), ("forecast", forecast)):
        monkeypatch.setattr(module, "main", recorder(name))
    return calls, codes


def step_names(calls):
    return [name for name, _ in calls]


def test_parse_args_defaults():
    args = pipeline.parse_args([])

    assert (args.train, args.explain, args.log_file, args.verbose) == ("auto", True, None, False)


@pytest.mark.parametrize(
    ("train_option", "model_exists", "expected"),
    [
        ("auto", False, ["ingest", "train", "forecast"]),
        ("auto", True, ["ingest", "forecast"]),
        ("always", True, ["ingest", "train", "forecast"]),
        ("never", False, ["ingest", "forecast"]),
    ],
)
def test_plan_steps_training_policy(train_option, model_exists, expected):
    args = pipeline.parse_args(["--train", train_option])

    steps = pipeline.plan_steps(args, model_exists=model_exists)

    assert [name for name, _, _ in steps] == expected


def test_plan_steps_passes_logging_to_every_step_and_no_explain_only_to_forecast(tmp_path):
    log_file = tmp_path / "pipeline.log"
    args = pipeline.parse_args(["--train", "always", "--no-explain", "--log-file", str(log_file), "-v"])

    argv = {name: step_argv for name, _, step_argv in pipeline.plan_steps(args, model_exists=True)}

    common = ["--log-file", str(log_file), "-v"]
    assert argv == {"ingest": common, "train": common, "forecast": [*common, "--no-explain"]}


def test_main_runs_the_steps_in_order(cli_env, recorded_steps):
    calls, _ = recorded_steps

    assert pipeline.main([]) == EXIT_OK

    assert step_names(calls) == ["ingest", "train", "forecast"]  # no model in `cli_env` yet


@pytest.mark.parametrize(
    ("failing", "code", "ran"),
    [
        ("ingest", EXIT_INPUT_ERROR, ["ingest"]),
        ("train", EXIT_FAILURE, ["ingest", "train"]),
        ("forecast", EXIT_FAILURE, ["ingest", "train", "forecast"]),
    ],
)
def test_main_stops_at_the_first_failing_step_and_returns_its_code(cli_env, recorded_steps, failing, code, ran):
    calls, codes = recorded_steps
    codes[failing] = code

    assert pipeline.main([]) == code

    assert step_names(calls) == ran


def test_main_invalid_configuration_returns_exit_2_without_running_anything(cli_env, recorded_steps, monkeypatch):
    calls, _ = recorded_steps
    monkeypatch.setenv("AI_PROVIDER", "nvidia")  # without NVIDIA_API_KEY
    monkeypatch.setenv("NVIDIA_API_KEY", "")

    assert pipeline.main([]) == EXIT_INPUT_ERROR

    assert calls == []


def test_full_pipeline_with_the_real_steps(cli_env, training_span_observations, fast_hgb, fake_weather, monkeypatch):
    training_span_observations.to_csv(cli_env / "energy_weather_es_hourly.csv", index=False)
    log_file = cli_env / "logs" / "pipeline.log"

    # First run: there is no model, so it trains
    assert pipeline.main(["--log-file", str(log_file)]) == EXIT_OK

    assert (cli_env / "models" / "model.joblib").exists()
    report = json.loads((cli_env / "reports" / "forecast_20260114T23.json").read_text(encoding="utf-8"))
    assert len(report["predictions"]) == 24
    log = log_file.read_text(encoding="utf-8")
    for message in ("Ingestion finished", "Training finished", "Forecast finished", "Pipeline finished"):
        assert message in log

    # Second run: the model exists, so `train` must not run (it would fail here)
    monkeypatch.setattr(train, "main", lambda argv: pytest.fail("train should not run when a model exists"))
    assert pipeline.main([]) == EXIT_OK
