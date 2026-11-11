"""Tests of `domain.modeling` and `domain.predictor` (loading and inference of the sklearn model)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from energy_forecast.domain import modeling
from energy_forecast.domain.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame
from energy_forecast.domain.modeling import (
    TRAIN_END,
    VALIDATION_END,
    SeasonalNaiveModel,
    TemporalSplit,
    build_candidates,
    build_hgb_pipeline,
    build_ridge_pipeline,
    build_tuned_candidates,
    temporal_split,
    tune_hgb,
    tune_ridge,
)
from energy_forecast.domain.validator import validate_observations

HOURS_PER_SET = 120


@pytest.fixture
def training_data(hourly_observations) -> tuple[pd.DataFrame, pd.Series]:
    """`(X, y)` of `hourly_observations` (48 rows after the warm-up, all in January 2025)."""
    valid, _ = validate_observations(hourly_observations)
    return build_training_frame(valid)


def synthetic_features(index: pd.DatetimeIndex, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """`X` with every feature and a `y` that depends on `lag_24` and the hour, indexed by `time_utc`."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(25_000, 3_000, size=(len(index), len(FEATURE_COLUMNS))), index=index,
                     columns=FEATURE_COLUMNS)
    X["hour"], X["weekday"], X["month"] = index.hour, index.day_of_week, index.month
    X[["is_holiday", "is_weekend", "covid_flag"]] = 0
    X["temperature_2m"] = rng.normal(15, 8, len(index))
    y = 0.8 * X["lag_24"] + 200 * X["hour"] + rng.normal(0, 100, len(index))
    return X, y.rename("load_mw")


@pytest.fixture
def split_data() -> tuple[pd.DataFrame, pd.Series]:
    """`HOURS_PER_SET` hours at the end of 2024, at the start of 2025 and at the start of 2026."""
    index = (
        pd.date_range(end=TRAIN_END, periods=HOURS_PER_SET, freq="h")
        .append(pd.date_range(TRAIN_END + pd.Timedelta(hours=1), periods=HOURS_PER_SET, freq="h"))
        .append(pd.date_range(VALIDATION_END + pd.Timedelta(hours=1), periods=HOURS_PER_SET, freq="h"))
        .rename("time_utc")
    )
    return synthetic_features(index)


# ---------------------------------------------------------------- baseline


def test_seasonal_naive_predicts_lag_168(training_data):
    X, y = training_data

    predictions = SeasonalNaiveModel().fit(X, y).predict(X)

    np.testing.assert_array_equal(predictions, X["lag_168"].to_numpy())


def test_seasonal_naive_without_lag_168_raises(training_data):
    X, y = training_data

    with pytest.raises(ValueError, match="lag_168"):
        SeasonalNaiveModel().fit(X.drop(columns="lag_168"), y)


def test_seasonal_naive_predict_before_fit_raises(training_data):
    X, _ = training_data

    with pytest.raises(NotFittedError):
        SeasonalNaiveModel().predict(X)


# ---------------------------------------------------------------- temporal split


def test_temporal_split_is_ordered_and_disjoint(split_data):
    X, y = split_data

    split = temporal_split(X, y)

    assert split.X_train.index.max() < split.X_val.index.min()
    assert split.X_val.index.max() < split.X_test.index.min()
    assert len(split.X_train) == len(split.X_val) == len(split.X_test) == HOURS_PER_SET
    assert split.X_train.index.equals(split.y_train.index)
    assert split.X_test.index.equals(split.y_test.index)


def test_temporal_split_boundaries():
    # Two hours around each boundary
    index = pd.DatetimeIndex(
        [TRAIN_END, TRAIN_END + pd.Timedelta(hours=1), VALIDATION_END, VALIDATION_END + pd.Timedelta(hours=1)],
        name="time_utc",
    )

    split = temporal_split(*synthetic_features(index))

    assert list(split.X_train.index) == [TRAIN_END]  # 31 Dec 2024 23:00 UTC is still train
    assert list(split.X_val.index) == [TRAIN_END + pd.Timedelta(hours=1), VALIDATION_END]
    assert list(split.X_test.index) == [VALIDATION_END + pd.Timedelta(hours=1)]


def test_temporal_split_does_not_depend_on_row_order(split_data):
    X, y = split_data
    shuffled = X.sample(frac=1, random_state=0)

    split = temporal_split(shuffled, y.loc[shuffled.index])

    assert split.X_train.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(split.X_test, temporal_split(X, y).X_test)


def test_temporal_split_train_val_concatenates_in_order(split_data):
    split = temporal_split(*split_data)

    assert len(split.X_train_val) == 2 * HOURS_PER_SET
    assert split.X_train_val.index.is_monotonic_increasing
    assert split.X_train_val.index.equals(split.y_train_val.index)


def test_temporal_split_with_an_empty_set_raises(training_data):
    # The fixture only covers January 2025: no train (<= 2024) or test (2026) data
    with pytest.raises(ValueError, match="train set is empty"):
        temporal_split(*training_data)


def test_temporal_split_with_different_indexes_raises(split_data):
    X, y = split_data

    with pytest.raises(ValueError, match="same `time_utc` index"):
        temporal_split(X, y.iloc[1:])


# ---------------------------------------------------------------- candidates


def test_build_candidates_returns_the_three_models():
    candidates = build_candidates()

    assert list(candidates) == ["naive", "ridge", "hgb"]
    assert isinstance(candidates["naive"], SeasonalNaiveModel)


@pytest.mark.parametrize("name", ["naive", "ridge", "hgb"])
def test_candidates_fit_and_predict(training_data, name):
    X, y = training_data
    model = build_candidates()[name]

    predictions = model.fit(X, y).predict(X)

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("build", [build_ridge_pipeline, build_hgb_pipeline], ids=["ridge", "hgb"])
def test_models_predict_with_unknown_lags(training_data, build):
    # At inference time a lag can be NaN (blackout in the last week): the model must still predict
    X, y = training_data
    model = build().fit(X, y)
    with_nan = X.copy()
    with_nan.loc[with_nan.index[0], ["lag_24", "rolling_mean_24"]] = np.nan

    assert np.isfinite(model.predict(with_nan)).all()


def test_ridge_preprocessing_uses_every_feature_once():
    preprocessing = build_ridge_pipeline().named_steps["preprocessing"]

    used = [column for _, _, columns in preprocessing.transformers for column in columns]

    assert sorted(used) == sorted(FEATURE_COLUMNS)


def test_ridge_learns_a_better_fit_than_the_mean(split_data):
    split = temporal_split(*split_data)
    model = build_ridge_pipeline().fit(split.X_train_val, split.y_train_val)

    error = np.abs(model.predict(split.X_test) - split.y_test).mean()
    error_of_mean = np.abs(split.y_train_val.mean() - split.y_test).mean()

    assert error < error_of_mean / 2


def test_hgb_uses_categorical_calendar_and_default_params():
    model = build_hgb_pipeline().named_steps["model"]

    assert list(model.categorical_features) == CATEGORICAL_FEATURES
    assert model.max_iter == 500 and model.learning_rate == 0.05 and model.max_leaf_nodes == 31


def test_hgb_does_not_use_early_stopping(training_data):
    # Early stopping would hold out random rows of a time series: the number of trees is tuned instead
    X, y = training_data
    model = build_hgb_pipeline(max_iter=20).fit(X, y).named_steps["model"]

    assert model.early_stopping is False
    assert model.n_iter_ == 20


def test_hgb_params_can_be_overridden():
    model = build_hgb_pipeline(learning_rate=0.2, max_iter=10).named_steps["model"]

    assert (model.learning_rate, model.max_iter, model.max_leaf_nodes) == (0.2, 10, 31)


def test_ridge_alpha_can_be_set():
    assert build_ridge_pipeline().named_steps["model"].alpha == modeling.RIDGE_ALPHA
    assert build_ridge_pipeline(alpha=10.0).named_steps["model"].alpha == 10.0


def test_tune_hgb_returns_an_unfitted_pipeline_from_the_grid(split_data, monkeypatch):
    monkeypatch.setattr(modeling, "HGB_PARAM_GRID", {"model__max_leaf_nodes": [7, 15], "model__max_iter": [20]})
    split = temporal_split(*split_data)

    model = tune_hgb(split)

    assert isinstance(model, Pipeline)
    assert model.named_steps["model"].max_leaf_nodes in (7, 15)
    assert model.named_steps["model"].max_iter == 20
    with pytest.raises(NotFittedError):  # the training service fits it like any other candidate
        model.predict(split.X_test)


def test_tune_ridge_chooses_a_small_alpha_for_a_clean_linear_signal(split_data, monkeypatch):
    # `y` is almost linear in the features: a huge penalty only hurts, so the search must avoid it
    monkeypatch.setattr(modeling, "RIDGE_PARAM_GRID", {"model__alpha": [0.1, 1e6]})

    model = tune_ridge(temporal_split(*split_data))

    assert model.named_steps["model"].alpha == 0.1


def test_build_tuned_candidates_returns_the_three_models(split_data, monkeypatch):
    monkeypatch.setattr(modeling, "HGB_PARAM_GRID", {"model__max_iter": [20]})
    monkeypatch.setattr(modeling, "RIDGE_PARAM_GRID", {"model__alpha": [1.0]})
    split = temporal_split(*split_data)

    candidates = build_tuned_candidates(split)

    assert list(candidates) == ["naive", "ridge", "hgb"]
    for model in candidates.values():
        assert model.fit(split.X_train_val, split.y_train_val).predict(split.X_test).shape == (HOURS_PER_SET,)


def test_temporal_split_is_a_frozen_dataclass(split_data):
    split = temporal_split(*split_data)

    assert isinstance(split, TemporalSplit)
    with pytest.raises(AttributeError):
        split.X_train = split.X_test


# ---------------------------------------------------------------- inference (step 3.2)


@pytest.mark.skip(reason="TODO (paso 3.2): implementar `predict_forecast`")
def test_predict_forecast_returns_24_hourly_predictions(small_trained_model, hourly_observations, weather_forecast_24h):
    # TODO: `predict_forecast` → `Forecast` with 24 consecutive `HourlyPrediction`s after the origin.
    ...
