import unittest
from pathlib import Path

from labor_baseline import compare, load_arff


ROOT = Path(__file__).resolve().parents[1]


class LaborBaselineTests(unittest.TestCase):
    def test_existing_arff_data(self):
        original = load_arff(ROOT / "labor_data_before_feature_selection.arff")
        selected = load_arff(ROOT / "labor_data_after_feature_selection.arff")
        result = compare(original, selected)
        self.assertEqual(result["rows"], 57)
        self.assertEqual(result["folds"], 5)
        self.assertEqual(result["all_features"]["predictors"], 16)
        self.assertEqual(result["weka_selected_features"]["predictors"], 3)
        for section in ("all_features", "weka_selected_features"):
            self.assertTrue(0 <= result[section]["roc_auc"] <= 1)
            self.assertTrue(0 <= result[section]["balanced_accuracy"] <= 1)

    def test_row_label_mismatch_rejected(self):
        original = load_arff(ROOT / "labor_data_before_feature_selection.arff")
        selected = load_arff(ROOT / "labor_data_after_feature_selection.arff").iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "matching labels"):
            compare(original, selected)

    def test_new_selected_feature_rejected(self):
        original = load_arff(ROOT / "labor_data_before_feature_selection.arff")
        selected = load_arff(ROOT / "labor_data_after_feature_selection.arff").rename(
            columns={"wage-increase-second-year": "unknown"}
        )
        with self.assertRaisesRegex(ValueError, "unknown attributes"):
            compare(original, selected)


if __name__ == "__main__":
    unittest.main()
