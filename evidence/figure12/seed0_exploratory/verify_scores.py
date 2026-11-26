"""Recompute every displayed number using only Python's standard library."""
from collections import Counter
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def compute(rows, expected):
    result = {}
    assert len(rows) == 522
    assert set(r['dataset'] for r in rows) == set(expected)
    for name, methods in expected.items():
        result[name] = {}
        ids = []
        for method, target in methods.items():
            cell = [r for r in rows if r['dataset'] == name and r['method'] == method]
            assert len(cell) == target['n']
            assert len({r['sample_id'] for r in cell}) == len(cell)
            assert {r['seed'] for r in cell} == ({0} if method == 'Modest-Align' else {None})
            ids.append({r['sample_id']: r['target'] for r in cell})
            classes = target['class_order']
            matrix = [[0] * len(classes) for _ in classes]
            for row in cell:
                scores = row['cosine_scores']
                assert row['class_order'] == classes and len(scores) == len(classes)
                assert all(math.isfinite(v) for v in scores)
                prediction = max(range(len(scores)), key=scores.__getitem__)
                assert prediction == row['prediction']
                assert 0 <= row['target'] < len(classes)
                matrix[row['target']][prediction] += 1
            assert matrix == target['confusion_matrix']
            assert all(sum(r) > 0 for r in matrix)
            correct = sum(matrix[i][i] for i in range(len(classes)))
            top1 = 100 * correct / len(cell)
            balanced = sum(100 * matrix[i][i] / sum(matrix[i]) for i in range(len(classes))) / len(classes)
            assert correct == target['correct']
            assert abs(top1 - target['accuracy_percent']) < 1e-10
            assert abs(balanced - target['balanced_accuracy_percent']) < 1e-10
            result[name][method] = {'n': len(cell), 'correct': correct,
                                   'top1_percent': top1, 'balanced_accuracy_percent': balanced}
        assert ids[0] == ids[1], 'Methods must use the same sample IDs and labels'
    assert {r['method'] for r in rows} == {'CLIP', 'Modest-Align'}
    return result


def main():
    rows = [json.loads(s) for s in (HERE / 'predictions.jsonl').read_text().splitlines()]
    expected = json.loads((HERE / 'expected_results.json').read_text())
    print(json.dumps({'status': 'PASS', 'predictions': len(rows),
                      'cells': compute(rows, expected)}, indent=2))


if __name__ == '__main__':
    main()
