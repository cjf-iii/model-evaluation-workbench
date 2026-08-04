# Model card

## Purpose

This workbench compares XGBoost and Random Forest regressors under one evaluation contract. It demonstrates held-out testing, cross-validation, residual inspection, model serialization, and SHAP attribution.

## Dataset

Every row is synthetic. Inputs describe annual spend, audience reach, activation channels, term length, renewal probability, deal type, market scope, and vertical. The target is a synthetic commercial-value score generated from documented nonlinear relationships plus random noise.

## Evaluation

- 80/20 held-out split;
- shuffled K-fold cross-validation;
- R-squared, MAE, RMSE, MAPE, and residual quantiles;
- mean absolute SHAP magnitude on a bounded held-out sample.

## Intended use

- demonstrate a reproducible evaluation workflow;
- compare model behavior under identical inputs and splits;
- inspect how model choice changes error and feature attribution.

## Limitations

- Synthetic performance does not predict accuracy on customer data.
- SHAP explains fitted model behavior, not causal effects.
- This project does not establish production latency, scale, or reliability.
- A real deployment would require data contracts, drift monitoring, access controls, and domain review.
