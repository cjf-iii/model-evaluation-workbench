"""Reproducible model evaluation for tabular regression."""

from .data import generate_deal_dataset
from .evaluation import evaluate_models

__all__ = ["evaluate_models", "generate_deal_dataset"]
