#!/usr/bin/env python3
"""Validate the evaluation evidence produced by the documented CLI run."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_NAMES = ("xgboost", "random_forest")


def main() -> None:
    artifacts = Path("artifacts")
    report = json.loads((artifacts / "evaluation_report.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(artifacts / "held_out_predictions.csv")
    source = pd.read_csv(artifacts / "synthetic_deals.csv")

    if report["dataset_rows"] != 500 or len(source) != 500:
        raise RuntimeError("the documented verification run requires exactly 500 synthetic rows")
    if report["train_rows"] != 400 or report["held_out_rows"] != 100:
        raise RuntimeError("the recorded 80/20 split is incomplete")
    if report["cross_validation_rows"] != report["train_rows"]:
        raise RuntimeError("cross-validation must remain inside the training partition")
    if len(predictions) != report["held_out_rows"]:
        raise RuntimeError("held-out predictions do not cover every test row")

    sample = source.drop(columns=["value_score"]).iloc[:1]
    summary: dict[str, float] = {}
    for name in MODEL_NAMES:
        metrics = report["models"][name]
        if metrics["r_squared"] <= 0.5:
            raise RuntimeError(f"{name} did not clear the synthetic held-out quality floor")

        importance_path = artifacts / f"{name}_shap_importance.csv"
        importance = pd.read_csv(importance_path)
        if importance.empty or importance["mean_absolute_shap"].isna().any():
            raise RuntimeError(f"{name} SHAP output is empty or invalid")

        # This joblib file was created immediately beforehand from this checkout.
        bundle = joblib.load(artifacts / f"{name}_bundle.joblib")
        transformed = bundle["preprocessor"].transform(sample)
        prediction = bundle["model"].predict(transformed)
        if len(prediction) != 1:
            raise RuntimeError(f"{name} persisted bundle could not score one row")
        summary[name] = round(float(metrics["r_squared"]), 4)

    print(json.dumps({"held_out_r_squared": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
