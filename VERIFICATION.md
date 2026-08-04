# Verification Record

Verified locally on August 4, 2026.

## Host

- macOS 26.6;
- Apple M3 Max, 40-core GPU;
- ARM64;
- Metal 4;
- Python 3.12.13.

## Result

The Metal preflight passed and all 3 unit tests passed. A separate 500-row smoke evaluation completed held-out testing, three-fold cross-validation, residual analysis, SHAP attribution, and model serialization for both estimators.

| Model | Held-out R² | MAE | CV MAE |
|---|---:|---:|---:|
| XGBoost | 0.6367 | 3.9657 | 3.6954 |
| Random Forest | 0.6330 | 3.8728 | 3.5257 |

## Reproduce

```bash
./scripts/verify_macos_metal.sh
uv sync --python 3.12
uv run python -m unittest discover -s tests -v
uv run model-eval --rows 500 --estimators 50 --cv-folds 3 --output artifacts
```
