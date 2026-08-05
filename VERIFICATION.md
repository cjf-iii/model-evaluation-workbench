# Verification Record

Recorded August 4, 2026.

## Release gate

```bash
uv sync --locked --group dev --group notebook
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
uv run --group notebook jupyter nbconvert --to notebook --execute notebooks/01_data_to_validation.ipynb --stdout > /dev/null
uv run model-eval --rows 500 --estimators 50 --cv-folds 3 --output artifacts
uv run python scripts/verify_artifacts.py
```

## Acceptance criteria

- one locked `uv` environment on Python 3.12;
- public CPU-based GitHub Actions workflow;
- deterministic synthetic-data and validation tests;
- every reviewer-notebook cell executes in the locked notebook environment;
- 80/20 held-out split with cross-validation confined to training rows;
- both estimators produce complete metrics, residuals, predictions, SHAP tables, and bundles;
- both saved bundles reload and score a source row;
- held-out R-squared above 0.50 for this documented synthetic verification run.

## Recorded result

All 4 tests and every reviewer-notebook cell passed, including SHAP output for both estimators. The full run generated 500 rows, reserved 100 for held-out evaluation, and confined three-fold cross-validation to the remaining 400 training rows.

| Model | Held-out R² | MAE | CV MAE |
|---|---:|---:|---:|
| XGBoost | 0.6367 | 3.9657 | 3.6438 |
| Random Forest | 0.6330 | 3.8728 | 3.4501 |

Both SHAP tables were populated, all 100 held-out predictions and residuals were preserved, and both locally generated joblib bundles reloaded and scored a source row.
