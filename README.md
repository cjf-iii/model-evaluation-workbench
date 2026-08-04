# Model Evaluation Workbench

A reproducible comparison of XGBoost and Random Forest using held-out tests, cross-validation, residual analysis, serialized model bundles, and SHAP explainability.

This repository demonstrates the résumé claim:

> Trained and evaluated XGBoost and Random Forest models using held-out tests, cross-validation, error analysis, and SHAP explainability.

## Evidence produced by one run

```text
artifacts/
├── synthetic_deals.csv
├── evaluation_report.json
├── held_out_predictions.csv
├── xgboost_bundle.joblib
├── random_forest_bundle.joblib
├── xgboost_shap_importance.csv
└── random_forest_shap_importance.csv
```

The report preserves the split sizes, seed, fold count, held-out metrics, cross-validation distribution, and residual quantiles. The prediction file keeps every held-out truth, prediction, and residual for independent inspection.

## Inputs

- annual spend and audience reach;
- activation-channel count and contract term;
- renewal probability;
- deal type, market scope, and industry vertical.

The included generator creates synthetic rows from documented relationships. It does not contain customer information.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

model-eval --rows 1500 --estimators 160 --cv-folds 4 --output artifacts
```

For a fast local check:

```bash
model-eval --rows 300 --estimators 30 --cv-folds 2 --skip-shap
```

## Evaluation design

```text
synthetic rows
    |
    +--> 80% train --------------------+
    |                                  |
    |                        fit preprocessing
    |                                  |
    |                    +-------------+-------------+
    |                    |                           |
    |                XGBoost                 Random Forest
    |                    |                           |
    +--> 20% held out --> metrics, residuals, predictions
                         SHAP attribution, model bundles

full dataset --> shuffled K-fold pipelines --> cross-validation MAE
```

Preprocessing is fitted inside each cross-validation fold. The held-out partition remains outside training and model selection.

## Test

```bash
./scripts/verify_macos_metal.sh
python -m unittest discover -s tests -v
```

The preflight fails unless the host is an ARM64 Mac with Metal support. XGBoost, Random Forest, cross-validation, and SHAP are CPU workloads on macOS; the repository states that boundary explicitly instead of presenting them as Metal-accelerated models.

Tests cover deterministic data generation, input constraints, two-model fitting, held-out quality, and the complete artifact contract.

See [VERIFICATION.md](VERIFICATION.md) for the recorded host, measured results, and reproduction commands.

## Claim boundary

The models run and produce inspectable evidence. Their scores describe synthetic data only. See [MODEL_CARD.md](MODEL_CARD.md) before reusing the results in another context.

## License

MIT
