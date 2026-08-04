"""Run a reproducible two-model evaluation and save the evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .data import generate_deal_dataset
from .evaluation import evaluate_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=160)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-shap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = generate_deal_dataset(rows=args.rows, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output / "synthetic_deals.csv", index=False)
    results = evaluate_models(
        dataset,
        args.output,
        estimators=args.estimators,
        cv_folds=args.cv_folds,
        seed=args.seed,
        compute_shap=not args.skip_shap,
    )
    ranking = sorted(results.items(), key=lambda item: item[1].mae)
    print(
        json.dumps(
            {
                "ranking_by_held_out_mae": [name for name, _ in ranking],
                "models": {name: asdict(metrics) for name, metrics in results.items()},
                "artifacts": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
