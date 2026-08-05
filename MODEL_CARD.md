# Model Card

## Purpose

This workbench compares XGBoost and Random Forest regressors under one evaluation contract. It demonstrates held-out testing, training-only cross-validation, residual inspection, model serialization, and SHAP attribution.

## Dataset

Every row is synthetic. Inputs describe annual spend, audience reach, activation channels, term length, renewal probability, deal type, market scope, and vertical. The target is a synthetic commercial-value score generated from documented nonlinear relationships plus random noise.

## Evaluation

- seeded 80/20 train and held-out split;
- shuffled K-fold cross-validation over only the 80% training partition;
- preprocessing refitted independently inside each fold;
- held-out R-squared, MAE, RMSE, MAPE, and residual quantiles;
- mean absolute SHAP magnitude over a bounded held-out sample;
- persisted-bundle reload and inference check.

## Intended use

- demonstrate a reproducible evaluation workflow;
- compare model behavior under identical inputs and splits;
- inspect how model choice changes errors and feature attribution.

## Limitations

- Synthetic performance does not predict accuracy on customer data.
- SHAP explains fitted model behavior, not causal effects.
- Fixed estimator configurations are compared; no hyperparameter-search claim is made.
- This project does not establish production latency, scale, drift resilience, or reliability.
- A real deployment would require data contracts, monitoring, access controls, domain review, and a decision-specific acceptance threshold.

## Artifact safety

The saved bundles use joblib/pickle-compatible serialization. Load only artifacts created locally or obtained from a trusted source.
