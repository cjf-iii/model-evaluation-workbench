"""Held-out evaluation, cross-validation, residual analysis, and SHAP."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


NUMERIC_FEATURES = [
    "annual_spend",
    "audience_reach",
    "activation_channels",
    "term_months",
    "renewal_probability",
]
CATEGORICAL_FEATURES = ["deal_type", "market", "vertical"]
TARGET = "value_score"


@dataclass(frozen=True)
class ModelMetrics:
    """Comparable held-out and cross-validation metrics."""

    r_squared: float
    mae: float
    rmse: float
    mape: float
    cross_validation_mae_mean: float
    cross_validation_mae_std: float
    residual_p05: float
    residual_median: float
    residual_p95: float


def build_preprocessor() -> ColumnTransformer:
    """Create a leakage-safe numeric and categorical transformer."""

    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_models(estimators: int = 160, seed: int = 42) -> dict[str, object]:
    """Return models with comparable capacity and deterministic seeds."""

    return {
        "xgboost": XGBRegressor(
            n_estimators=estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=estimators,
            max_depth=16,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def _metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    cv_mae: np.ndarray,
) -> ModelMetrics:
    residuals = truth - predictions
    return ModelMetrics(
        r_squared=float(r2_score(truth, predictions)),
        mae=float(mean_absolute_error(truth, predictions)),
        rmse=float(mean_squared_error(truth, predictions) ** 0.5),
        mape=float(mean_absolute_percentage_error(truth, predictions)),
        cross_validation_mae_mean=float(np.mean(cv_mae)),
        cross_validation_mae_std=float(np.std(cv_mae)),
        residual_p05=float(np.quantile(residuals, 0.05)),
        residual_median=float(np.median(residuals)),
        residual_p95=float(np.quantile(residuals, 0.95)),
    )


def _shap_importance(
    model: object,
    transformed_sample: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return mean absolute SHAP magnitude for one fitted tree model."""

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(transformed_sample))
    if values.ndim != 2:
        raise ValueError(f"expected two-dimensional SHAP values, received shape {values.shape}")
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_absolute_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_models(
    dataset: pd.DataFrame,
    output_dir: str | Path,
    *,
    estimators: int = 160,
    cv_folds: int = 4,
    seed: int = 42,
    compute_shap: bool = True,
) -> dict[str, ModelMetrics]:
    """Fit both models and write auditable evaluation artifacts."""

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    features = dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = dataset[TARGET].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=seed,
    )

    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    feature_names = list(preprocessor.get_feature_names_out())
    models = build_models(estimators=estimators, seed=seed)
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    all_metrics: dict[str, ModelMetrics] = {}
    prediction_frame = pd.DataFrame({"row_index": x_test.index, "actual": y_test})

    for name, model in models.items():
        model.fit(transformed_train, y_train)
        predictions = np.asarray(model.predict(transformed_test))

        pipeline = Pipeline(
            [
                ("preprocessor", clone(build_preprocessor())),
                ("model", clone(model)),
            ]
        )
        cv_mae = -cross_val_score(
            pipeline,
            features,
            target,
            cv=splitter,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        all_metrics[name] = _metrics(y_test, predictions, cv_mae)
        prediction_frame[f"{name}_prediction"] = predictions
        prediction_frame[f"{name}_residual"] = y_test - predictions

        if compute_shap:
            sample = transformed_test[: min(200, len(transformed_test))]
            importance = _shap_importance(model, sample, feature_names)
            importance.to_csv(destination / f"{name}_shap_importance.csv", index=False)

        joblib.dump(
            {
                "preprocessor": preprocessor,
                "model": model,
                "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "target": TARGET,
            },
            destination / f"{name}_bundle.joblib",
        )

    prediction_frame.to_csv(destination / "held_out_predictions.csv", index=False)
    report = {
        "dataset_rows": len(dataset),
        "train_rows": len(x_train),
        "held_out_rows": len(x_test),
        "seed": seed,
        "cv_folds": cv_folds,
        "models": {name: asdict(metrics) for name, metrics in all_metrics.items()},
    }
    (destination / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return all_metrics
