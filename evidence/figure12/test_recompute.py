"""Run with python -m unittest -v test_recompute.py."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from recompute import ROOT, calculate_cell, load_records, verify_and_recompute


class ArithmeticChecks(unittest.TestCase):
    def test_all_released_cells(self):
        result = verify_and_recompute()
        self.assertEqual(result['verification'], 'PASS')
        self.assertEqual(len(result['cells']), 35)
        self.assertEqual(result['records'], 1827)
        self.assertEqual(len(result['legacy_rows']), 36)
        self.assertFalse(result['historical_five_value_aggregation_reconstructed'])

    def setUp(self):
        records = load_records(ROOT / 'predictions.jsonl')
        self.records = [r for r in records if r['dataset'] == 'CheXpert' and r['method'] == 'CLIP']
        self.expected = json.loads((ROOT / 'protocol.json').read_text())['datasets']['CheXpert']

    def test_missing_sample_fails(self):
        with self.assertRaises(ValueError):
            calculate_cell(self.records[:-1], self.expected, 'CheXpert', 'CLIP', None)

    def test_wrong_argmax_fails(self):
        rows = copy.deepcopy(self.records)
        rows[0]['prediction'] = 1 - rows[0]['prediction']
        with self.assertRaises(ValueError):
            calculate_cell(rows, self.expected, 'CheXpert', 'CLIP', None)

    def test_label_index_mismatch_fails(self):
        rows = copy.deepcopy(self.records)
        rows[0]['target'] = 1 - rows[0]['target']
        with self.assertRaises(ValueError):
            calculate_cell(rows, self.expected, 'CheXpert', 'CLIP', None)

    def test_nonfinite_score_fails(self):
        rows = copy.deepcopy(self.records)
        rows[0]['cosine_scores'][0] = float('nan')
        with self.assertRaises(ValueError):
            calculate_cell(rows, self.expected, 'CheXpert', 'CLIP', None)

    def test_duplicate_class_fails(self):
        rows = copy.deepcopy(self.records)
        rows[0]['class_order'][1] = rows[0]['class_order'][0]
        with self.assertRaises(ValueError):
            calculate_cell(rows, self.expected, 'CheXpert', 'CLIP', None)

    def test_duplicate_sample_fails(self):
        rows = copy.deepcopy(self.records)
        rows[1]['sample_id'] = rows[0]['sample_id']
        with self.assertRaises(ValueError):
            calculate_cell(rows, self.expected, 'CheXpert', 'CLIP', None)


if __name__ == '__main__':
    unittest.main()
