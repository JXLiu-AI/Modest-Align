import copy
import json
from pathlib import Path
import unittest
from verify_scores import compute


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        cls.rows = [json.loads(s) for s in (root / 'predictions.jsonl').read_text().splitlines()]
        cls.expected = json.loads((root / 'expected_results.json').read_text())

    def test_complete_release(self):
        self.assertEqual(len(compute(self.rows, self.expected)), 5)

    def test_missing_row_rejected(self):
        with self.assertRaises(AssertionError):
            compute(self.rows[:-1], self.expected)

    def test_nonfinite_score_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['cosine_scores'][0] = float('nan')
        with self.assertRaises(AssertionError):
            compute(rows, self.expected)

    def test_forged_prediction_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['prediction'] = 1 - rows[0]['prediction']
        with self.assertRaises(AssertionError):
            compute(rows, self.expected)

    def test_changed_seed_rejected(self):
        rows = copy.deepcopy(self.rows)
        next(r for r in rows if r['method'] == 'Modest-Align')['seed'] = 42
        with self.assertRaises(AssertionError):
            compute(rows, self.expected)

    def test_unpaired_identifiers_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['sample_id'] += '-wrong'
        with self.assertRaises(AssertionError):
            compute(rows, self.expected)


if __name__ == '__main__':
    unittest.main()
