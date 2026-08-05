from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_evaluation_workbench.data import generate_deal_dataset
from model_evaluation_workbench.evaluation import evaluate_models


class EvaluationTests(unittest.TestCase):
    def test_evaluation_writes_auditable_artifacts(self) -> None:
        dataset = generate_deal_dataset(rows=240, seed=11)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            metrics = evaluate_models(
                dataset,
                output,
                estimators=50,
                cv_folds=2,
                seed=11,
                compute_shap=False,
            )
            self.assertEqual(set(metrics), {"xgboost", "random_forest"})
            for result in metrics.values():
                self.assertGreater(result.r_squared, 0.5)
                self.assertGreater(result.mae, 0)

            report = json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["dataset_rows"], 240)
            self.assertEqual(report["train_rows"] + report["held_out_rows"], 240)
            self.assertEqual(report["cross_validation_rows"], report["train_rows"])
            self.assertTrue((output / "held_out_predictions.csv").is_file())
            self.assertTrue((output / "xgboost_bundle.joblib").is_file())
            self.assertTrue((output / "random_forest_bundle.joblib").is_file())

    def test_shap_outputs_cover_both_models(self) -> None:
        dataset = generate_deal_dataset(rows=160, seed=17)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            evaluate_models(
                dataset,
                output,
                estimators=20,
                cv_folds=2,
                seed=17,
                compute_shap=True,
            )
            for model_name in ("xgboost", "random_forest"):
                path = output / f"{model_name}_shap_importance.csv"
                self.assertTrue(path.is_file())
                self.assertIn("mean_absolute_shap", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
