from __future__ import annotations

import unittest

import pandas as pd

from model_evaluation_workbench.data import generate_deal_dataset


class DatasetTests(unittest.TestCase):
    def test_generation_is_deterministic_and_complete(self) -> None:
        first = generate_deal_dataset(rows=120, seed=7)
        second = generate_deal_dataset(rows=120, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 120)
        self.assertTrue(first["value_score"].between(1, 100).all())

    def test_too_few_rows_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_deal_dataset(rows=50)


if __name__ == "__main__":
    unittest.main()
