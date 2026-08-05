"""Deterministic synthetic data with documented commercial inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEAL_TYPES = np.array(["broadcast", "digital", "experiential", "integrated"])
MARKETS = np.array(["local", "regional", "national", "global"])
VERTICALS = np.array(["auto", "finance", "gaming", "retail", "technology"])


def generate_deal_dataset(rows: int = 1_500, seed: int = 42) -> pd.DataFrame:
    """Generate a nonlinear regression dataset for model comparison.

    The target is a synthetic commercial-value score. Inputs include deal
    economics, audience scale, activation breadth, term, and categorical
    market context. No row describes a real customer or transaction.
    """

    if rows < 100:
        raise ValueError("rows must be at least 100")

    rng = np.random.default_rng(seed)
    annual_spend = np.exp(rng.normal(np.log(600_000), 1.0, rows)).clip(25_000, 12_000_000)
    audience_reach = np.exp(rng.normal(np.log(2_000_000), 1.05, rows)).clip(50_000, 80_000_000)
    activation_channels = rng.integers(1, 7, rows)
    term_months = rng.choice(
        [3, 6, 12, 18, 24, 36], size=rows, p=[0.08, 0.12, 0.32, 0.12, 0.24, 0.12]
    )
    renewal_probability = rng.beta(5, 2, rows)
    deal_type = rng.choice(DEAL_TYPES, size=rows, p=[0.18, 0.25, 0.12, 0.45])
    market = rng.choice(MARKETS, size=rows, p=[0.2, 0.25, 0.42, 0.13])
    vertical = rng.choice(VERTICALS, size=rows)

    deal_effect = (
        pd.Series(deal_type)
        .map({"broadcast": 2.0, "digital": 3.5, "experiential": 4.5, "integrated": 8.0})
        .to_numpy()
    )
    market_effect = (
        pd.Series(market)
        .map({"local": 0.0, "regional": 2.5, "national": 6.0, "global": 7.5})
        .to_numpy()
    )
    vertical_effect = (
        pd.Series(vertical)
        .map({"auto": 3.0, "finance": 5.0, "gaming": 4.0, "retail": 1.5, "technology": 6.0})
        .to_numpy()
    )

    value_score = (
        7.0
        + 5.2 * np.log10(annual_spend)
        + 2.8 * np.log10(audience_reach)
        + 2.4 * activation_channels
        + 0.18 * term_months
        + 13.0 * renewal_probability
        + deal_effect
        + market_effect
        + vertical_effect
        + 2.2 * (deal_type == "integrated") * activation_channels
        + rng.normal(0, 4.0, rows)
    ).clip(1, 100)

    return pd.DataFrame(
        {
            "annual_spend": annual_spend.round(2),
            "audience_reach": audience_reach.astype(int),
            "activation_channels": activation_channels,
            "term_months": term_months,
            "renewal_probability": renewal_probability.round(4),
            "deal_type": deal_type,
            "market": market,
            "vertical": vertical,
            "value_score": value_score.round(4),
        }
    )
