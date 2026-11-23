"""Tests for `logging_config`."""

import logging

import pytest

from energy_forecast.logging_config import LOG_FORMAT, get_logger, setup_logging


pytestmark = pytest.mark.usefixtures("restore_root_logger")


def test_sets_level_and_format():
    setup_logging("DEBUG")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert root.handlers[0].formatter._fmt == LOG_FORMAT


def test_calling_twice_does_not_duplicate_handlers():
    setup_logging()
    setup_logging()

    assert len(logging.getLogger().handlers) == 1


def test_writes_to_log_file_creating_its_folder(tmp_path):
    log_file = tmp_path / "logs" / "run.log"

    setup_logging(log_file=log_file)
    get_logger("test").info("hello log")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "INFO | energy_forecast.test | hello log" in content


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        setup_logging("NOT_A_LEVEL")


def test_get_logger_uses_project_namespace():
    assert get_logger("train").name == "energy_forecast.train"
