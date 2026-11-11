"""Candidate models and temporal split.

All candidates share the sklearn interface (`fit` / `predict`) and receive `X[FEATURE_COLUMNS]` from
`build_training_frame`, so the training service compares them without knowing which one is which.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from energy_forecast.domain.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS

# Temporal split (never random): train 2019-2024, validation 2025, test 2026
TRAIN_END = pd.Timestamp("2024-12-31T23:00:00+00:00")
VALIDATION_END = pd.Timestamp("2025-12-31T23:00:00+00:00")

RANDOM_STATE = 42
# No early stopping: it would hold out a random 10 % of rows, whose neighbouring hours stay in the training
# data (a random split of a time series). The number of trees is tuned with `TimeSeriesSplit` instead.
HGB_PARAMS = {
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "early_stopping": False,
}
RIDGE_ALPHA = 1.0

# Reduced grids for `tune_*` (cross-validated with `TimeSeriesSplit` on train + validation)
HGB_PARAM_GRID = {
    "model__learning_rate": [0.05, 0.1],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__max_iter": [500, 1000],
}
RIDGE_PARAM_GRID = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
TUNING_SPLITS = 3

NAIVE_FEATURE = "lag_168"
TEMPERATURE_FEATURE = "temperature_2m"
BINARY_FEATURES = ["is_holiday", "is_weekend", "covid_flag"]
# Numeric features scaled for Ridge: lags, rolling means and the weather except the temperature (it goes to splines)
SCALED_FEATURES = [
    column
    for column in FEATURE_COLUMNS
    if column not in {*CATEGORICAL_FEATURES, *BINARY_FEATURES, TEMPERATURE_FEATURE}
]


@dataclass(frozen=True)
class TemporalSplit:
    """Train / validation / test sets, ordered in time."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def X_train_val(self) -> pd.DataFrame:
        """Train + validation: what the final model is fitted on before being evaluated on test."""
        return pd.concat([self.X_train, self.X_val])

    @property
    def y_train_val(self) -> pd.Series:
        return pd.concat([self.y_train, self.y_val])


class SeasonalNaiveModel(BaseEstimator, RegressorMixin):
    """Baseline: predicts the demand of the same hour of the previous week (`lag_168`)."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SeasonalNaiveModel":
        """Learns nothing; it exists to be compared through the same interface as the rest."""
        if NAIVE_FEATURE not in X.columns:
            raise ValueError(f"SeasonalNaiveModel needs the '{NAIVE_FEATURE}' column")
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns `lag_168` as the prediction."""
        check_is_fitted(self)
        return X[NAIVE_FEATURE].to_numpy(dtype=float)


def temporal_split(X: pd.DataFrame, y: pd.Series) -> TemporalSplit:
    """Splits by date: train up to `TRAIN_END`, validation up to `VALIDATION_END` and test afterwards.

    `X` and `y` are indexed by `time_utc` (output of `build_training_frame`).
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must have the same `time_utc` index")
    X, y = X.sort_index(), y.sort_index()
    train = X.index <= TRAIN_END
    val = (X.index > TRAIN_END) & (X.index <= VALIDATION_END)
    test = X.index > VALIDATION_END

    for name, mask in (("train", train), ("validation", val), ("test", test)):
        if not mask.any():
            raise ValueError(
                f"The {name} set is empty: the data covers {X.index.min()} - {X.index.max()} "
                f"(train <= {TRAIN_END}, validation <= {VALIDATION_END}, test after)"
            )
    return TemporalSplit(X[train], y[train], X[val], y[val], X[test], y[test])


def build_ridge_pipeline(alpha: float = RIDGE_ALPHA) -> Pipeline:
    """Interpretable linear model: one-hot of the calendar + temperature splines + `Ridge`.

    - `hour`, `weekday`, `month` → one-hot: a linear model cannot learn that hour 23 is not "23 times hour 1".
    - `temperature_2m` → splines: demand grows both with cold and with heat (U-shaped relation).
    - Lags, rolling means and the rest of the weather → median imputation + scaling (Ridge needs no NaN
      and comparable scales for its penalty).
    - `is_holiday`, `is_weekend`, `covid_flag` → as they are (already 0/1).
    """
    preprocessing = ColumnTransformer(
        [
            ("calendar", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            (
                "temperature",
                make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=6, degree=3)),
                [TEMPERATURE_FEATURE],
            ),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), SCALED_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", Ridge(alpha=alpha))])


def build_hgb_pipeline(**params) -> Pipeline:
    """Final model: `HistGradientBoostingRegressor` with `HGB_PARAMS` (overridable) and categorical calendar.

    No imputation or scaling: HGB accepts NaN natively and trees do not depend on the scale.
    """
    model = HistGradientBoostingRegressor(
        categorical_features=CATEGORICAL_FEATURES,
        random_state=RANDOM_STATE,
        **{**HGB_PARAMS, **params},
    )
    return Pipeline([("model", model)])


def build_candidates() -> dict[str, BaseEstimator]:
    """The three models to compare: `naive`, `ridge` and `hgb`."""
    return {"naive": SeasonalNaiveModel(), "ridge": build_ridge_pipeline(), "hgb": build_hgb_pipeline()}


def _grid_search(pipeline: Pipeline, param_grid: dict, split: TemporalSplit) -> Pipeline:
    """Best `param_grid` combination by MAE with `TimeSeriesSplit` on train + validation, as an UNFITTED pipeline.

    `TimeSeriesSplit` always validates on data after the training folds, as in production. The test set
    is not used, so its metrics stay honest. The pipeline is returned unfitted (`refit=False`), like the ones
    of `build_candidates`: the training service fits every candidate in the same way.
    """
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=TimeSeriesSplit(n_splits=TUNING_SPLITS),
        scoring="neg_mean_absolute_error",
        refit=False,
    )
    search.fit(split.X_train_val, split.y_train_val)
    return clone(pipeline).set_params(**search.best_params_)


def tune_hgb(split: TemporalSplit) -> Pipeline:
    """HGB with the best `learning_rate`, `max_leaf_nodes` and `max_iter` of `HGB_PARAM_GRID`."""
    return _grid_search(build_hgb_pipeline(), HGB_PARAM_GRID, split)


def tune_ridge(split: TemporalSplit) -> Pipeline:
    """Ridge with the best `alpha` of `RIDGE_PARAM_GRID`."""
    return _grid_search(build_ridge_pipeline(), RIDGE_PARAM_GRID, split)


def build_tuned_candidates(split: TemporalSplit) -> dict[str, BaseEstimator]:
    """Like `build_candidates`, with the hyperparameters of Ridge and HGB tuned (`train --tune`)."""
    return {"naive": SeasonalNaiveModel(), "ridge": tune_ridge(split), "hgb": tune_hgb(split)}
