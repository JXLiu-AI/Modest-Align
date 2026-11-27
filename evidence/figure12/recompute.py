"""Recompute all released arithmetic from scores; no ML framework required."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
SEEDS = [0, 1, 3, 7, 42]
DATASETS = ['CheXpert', 'Shenzhen', 'BrainTumor', 'IDRiD', 'D&C']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def calculate_cell(records, expected, dataset, method, seed):
    require(len(records) == expected['n'], 'Missing/extra samples in {}'.format((dataset, method, seed)))
    require(len({r['sample_id'] for r in records}) == len(records), 'Duplicate sample ID')
    target_counts = Counter()
    correct_counts = Counter()
    prediction_counts = Counter()
    images = []
    for row in records:
        classes = row['class_order']
        scores = row['cosine_scores']
        require(len(classes) == len(set(classes)) == len(scores), 'Invalid/duplicate class candidates')
        require(all(math.isfinite(v) and abs(v) <= 1.00001 for v in scores), 'Invalid cosine score')
        require(0 <= row['target'] < len(classes), 'Invalid target')
        require(classes[row['target']] == row['label'], 'Label/index disagreement')
        predicted = max(range(len(scores)), key=scores.__getitem__)
        require(predicted == row['prediction'], 'Stored prediction does not equal argmax')
        target = row['label']
        prediction = classes[predicted]
        hit = int(target == prediction)
        target_counts[target] += 1
        correct_counts[target] += hit
        prediction_counts[prediction] += 1
        images.append({'dataset': dataset, 'method': method, 'seed': seed,
                       'sample_id': row['sample_id'], 'target_class': target,
                       'predicted_class': prediction, 'correct': hit,
                       'top1_cosine': scores[predicted],
                       'scores_by_class': json.dumps(dict(zip(classes, scores)), sort_keys=True)})
    require(dict(target_counts) == expected['class_counts'], 'Wrong class composition')
    n = len(records)
    correct = sum(correct_counts.values())
    accuracy = 100 * correct / n
    classes = sorted(target_counts)
    per_class = [{'dataset': dataset, 'method': method, 'seed': seed, 'class': name,
                  'n': target_counts[name], 'correct': correct_counts[name],
                  'recall_percent': 100 * correct_counts[name] / target_counts[name]}
                 for name in classes]
    balanced = statistics.mean(r['recall_percent'] for r in per_class)
    return ({'dataset': dataset, 'method': method, 'seed': seed, 'n': n,
             'correct': correct, 'accuracy_percent': accuracy,
             'balanced_accuracy_percent': balanced,
             'prediction_counts': dict(prediction_counts)}, images, per_class)


def verify_and_recompute(root=ROOT):
    protocol = read_json(root / 'protocol.json')
    expected_results = read_json(root / 'expected_results.json')
    records = load_records(root / 'predictions.jsonl')
    groups = defaultdict(list)
    labels_by_sample = {}
    for row in records:
        key = (row['dataset'], row['method'], row['seed'])
        groups[key].append(row)
        sample_key = (row['dataset'], row['sample_id'])
        require(labels_by_sample.setdefault(sample_key, row['label']) == row['label'],
                'Methods do not share the same sample labels')
    expected_cells = {(name, 'CLIP', None) for name in DATASETS}
    expected_cells |= {(name, 'COCO_RP_ES', seed) for name in DATASETS for seed in SEEDS}
    expected_cells |= {(name, 'candidate_20240906195', 42) for name in DATASETS}
    require(set(groups) == expected_cells, 'Cells/seeds are missing or were added')
    cells, per_image, per_class = [], [], []
    expected_sample_ids = {}
    for dataset in DATASETS:
        for method, seed in [('CLIP', None)] + [('COCO_RP_ES', s) for s in SEEDS] + [('candidate_20240906195', 42)]:
            cell_records = groups[(dataset, method, seed)]
            sample_ids = {r['sample_id'] for r in cell_records}
            require(expected_sample_ids.setdefault(dataset, sample_ids) == sample_ids,
                    'Methods/seeds do not use identical sample IDs')
            cell, images, classes = calculate_cell(cell_records, protocol['datasets'][dataset], dataset, method, seed)
            cells.append(cell)
            per_image.extend(images)
            per_class.extend(classes)
            expected = expected_results['cells'][dataset][method][str(seed)]
            for key in ('accuracy_percent', 'balanced_accuracy_percent'):
                require(abs(cell[key] - expected[key]) < 2e-5, 'Published new-evaluation summary disagrees')
    aggregates = []
    for dataset in DATASETS:
        selected = [c for c in cells if c['dataset'] == dataset and c['method'] == 'COCO_RP_ES']
        require([c['seed'] for c in selected] == SEEDS, 'Wrong seed order/set')
        for metric in ('accuracy_percent', 'balanced_accuracy_percent'):
            values = [c[metric] for c in selected]
            mean = statistics.mean(values)
            squares = [(value - mean) ** 2 for value in values]
            sd = math.sqrt(math.fsum(squares) / (len(values) - 1))
            require(abs(sd - statistics.stdev(values)) < 1e-10, 'SD arithmetic disagreement')
            old = expected_results['coco_aggregates'][dataset][metric]
            require(abs(mean - old['mean']) < 2e-5 and abs(sd - old['sd']) < 2e-5,
                    'Seed aggregate does not reproduce the supplied result')
            aggregates.append({'dataset': dataset, 'metric': metric, 'seeds': SEEDS,
                               'values': values, 'mean': mean, 'squared_deviations': squares,
                               'sum_squared_deviations': math.fsum(squares), 'sample_sd': sd})

    with (root / 'legacy_metric_rows.csv').open(newline='') as stream:
        legacy = list(csv.DictReader(stream))
    require(len(legacy) == 36, 'Incomplete surviving legacy ledger')
    for row in legacy:
        for key in ('txt_r1', 'txt_r5', 'txt_r10', 'txt_r_mean',
                    'img_r1', 'img_r5', 'img_r10', 'img_r_mean', 'r_mean', 'agg_metrics'):
            row[key] = float(row[key])
        row['calculated_txt_mean'] = statistics.mean(row['txt_r' + str(k)] for k in (1, 5, 10))
        row['calculated_img_mean'] = statistics.mean(row['img_r' + str(k)] for k in (1, 5, 10))
        row['calculated_r_mean'] = (row['calculated_txt_mean'] + row['calculated_img_mean']) / 2
        for calculated, recorded in [('calculated_txt_mean', 'txt_r_mean'),
                                     ('calculated_img_mean', 'img_r_mean'), ('calculated_r_mean', 'r_mean'),
                                     ('calculated_txt_mean', 'agg_metrics')]:
            require(abs(row[calculated] - row[recorded]) < 1e-8, 'Legacy formula disagrees')

    fingerprint = read_json(root / 'text_cache_fingerprint.json')
    cached = fingerprint['vectors']['cached']
    chars = fingerprint['vectors']['fresh_characters']
    whole = fingerprint['vectors']['fresh_whole_caption'][0]

    def cosine(a, b):
        return math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(
            math.fsum(x * x for x in a) * math.fsum(y * y for y in b))

    require(len(cached) == 48 and len(chars) == 16, 'Wrong text-cache dimensions')
    fp_rows = []
    for i, vector in enumerate(cached):
        character = chars[i % 16]
        require(len(vector) == len(character) == len(whole) == 1024, 'Wrong embedding dimension')
        require(all(math.isfinite(v) for v in vector + character + whole), 'Nonfinite fingerprint vector')
        difference = max(abs(a - b) for a, b in zip(vector, character))
        close = all(abs(a - b) <= 2e-5 + 2e-5 * abs(b) for a, b in zip(vector, character))
        char_cosine = cosine(vector, character)
        whole_cosine = cosine(vector, whole)
        require(close and char_cosine > 0.999999 and whole_cosine < 0.7, 'Text-cache identity check fails')
        fp_rows.append({'cache_index': i, 'mapped_image_index': i // 16,
                        'processed_character': fingerprint['processed_characters'][i % 16],
                        'character_cosine': char_cosine, 'whole_caption_cosine': whole_cosine,
                        'max_abs_difference': difference, 'allclose': close})

    with (root / 'historical_reported_values.csv').open(newline='') as stream:
        historical = list(csv.DictReader(stream))
    for row in historical:
        n = protocol['datasets'][row['dataset']]['n']
        for method in ('CLIP', 'Modest-Align'):
            value = float(row[method])
            row[method + '_compatible_with_single_top1_on_current_list'] = any(
                round(100 * k / n, 2) == value for k in range(n + 1))
            row[method + '_compatible_with_three_equal_50_image_top1_runs'] = any(
                round(100 * k / 150, 2) == value for k in range(151))
        row['current_list_n_not_verified_historical_n'] = n
    require(sum(p['n'] for p in protocol['datasets'].values()) == 261, 'Wrong total image count')
    return {'verification': 'PASS', 'historical_five_value_aggregation_reconstructed': False,
            'scope': 'Full arithmetic reproduction from released scores, not pixel-level model inference.',
            'records': len(records), 'cells': cells, 'coco_aggregates': aggregates,
            'legacy_rows': legacy, 'text_fingerprint_rows': fp_rows,
            'historical_arithmetic_checks': historical,
            'per_image': per_image, 'per_class': per_class}


def write_report(result, output):
    output.mkdir(exist_ok=True, parents=True)
    write_json(output / 'verified_results.json', {k: v for k, v in result.items() if k != 'per_image'})
    write_csv(output / 'per_image_calculations.csv', result['per_image'])
    write_csv(output / 'per_class_calculations.csv', result['per_class'])
    write_csv(output / 'legacy_recalculated.csv', result['legacy_rows'])
    write_csv(output / 'text_fingerprint_calculations.csv', result['text_fingerprint_rows'])
    text = ['# Complete Arithmetic Record', '',
            'Generated by `python recompute.py`. Every cell and seed is included.', '',
            '**Scope:** score-to-metric reproduction; not a reconstruction of the historical five percentages.', '',
            '## 1. Per-image decisions', '',
            '`prediction = first argmax(cosine_scores)`; `correct = int(predicted_class == target_class)`.',
            'Every image decision and all class scores appear in `per_image_calculations.csv`.', '',
            '## 2. Counts and class-level metrics', '',
            '`top1_percent = 100 * correct / n`.',
            '`balanced_accuracy_percent = mean(100 * class_correct / class_n)`.',
            'Per-class numerators/denominators are in `per_class_calculations.csv`.', '',
            '| Dataset | Evaluation | Seed | Correct / N | Top-1 (%) | Balanced (%) |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for cell in result['cells']:
        text.append('| {dataset} | {method} | {seed} | {correct}/{n} | {accuracy_percent:.8f} | {balanced_accuracy_percent:.8f} |'.format(**cell))
    text += ['', '## 3. All five COCO adapter seeds', '',
             '`mean = sum(values) / 5`.',
             '`sample_sd = sqrt(sum((value - mean)^2) / 4)`; this is across adapters, not image resampling.', '']
    for row in result['coco_aggregates']:
        text += ['### {}: {}'.format(row['dataset'], row['metric']), '',
                 '- Seeds: `{}`.'.format(row['seeds']),
                 '- Values (%): `{}`.'.format([round(v, 10) for v in row['values']]),
                 '- Mean: `{:.10f}`.'.format(row['mean']),
                 '- Squared deviations: `{}`.'.format([round(v, 10) for v in row['squared_deviations']]),
                 '- Sum of squared deviations: `{:.10f}`.'.format(row['sum_squared_deviations']),
                 '- Sample SD: `sqrt({:.10f}/4) = {:.10f}`.'.format(row['sum_squared_deviations'], row['sample_sd']), '']
    text += ['## 4. Every surviving retrieval metric row', '',
             '`txt_r_mean = (txt_r1 + txt_r5 + txt_r10) / 3`.',
             '`img_r_mean = (img_r1 + img_r5 + img_r10) / 3`.',
             '`r_mean = (txt_r_mean + img_r_mean) / 2`; `agg_metrics = txt_r_mean`.',
             'These formulas do not by themselves define class-level classification accuracy.', '',
             '| Output bucket | Line | txt R@1 | txt R@5 | txt R@10 | Calculated mean | Recorded agg_metrics |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['legacy_rows']:
        text.append('| {run_id} | {line} | {txt_r1:.8f} | {txt_r5:.8f} | {txt_r10:.8f} | {calculated_txt_mean:.8f} | {agg_metrics:.8f} |'.format(**row))
    text += ['', '## 5. Historical percentages: limited arithmetic tests', '',
             'The current list sizes are NOT established as the historical denominators.',
             'The authors\' account of approximately 50 images, a multi-seed Modest-Align mean and a single CLIP run is separately attributed in [HISTORICAL_AUTHOR_CLARIFICATION.md](HISTORICAL_AUTHOR_CLARIFICATION.md).',
             'The 3 x 50 column tests exactly three equal-weight runs of exactly 50 image queries each. Neither this exact seed count nor these exact denominators are established by that account.',
             'A failed test rejects only the specified counting hypothesis, not every possible aggregation.',
             'No grouping or averaging formula was invented to fit a published percentage.', '',
             '| Dataset | Historical CLIP | Historical Modest-Align | Current-list N | Modest compatible with one top-1 count on this list? | Modest compatible with 3 x 50 top-1 mean? |',
             '| --- | ---: | ---: | ---: | --- | --- |']
    for row in result['historical_arithmetic_checks']:
        text.append('| {} | {} | {} | {} | {} | {} |'.format(
            row['dataset'], row['CLIP'], row['Modest-Align'], row['current_list_n_not_verified_historical_n'],
            row['Modest-Align_compatible_with_single_top1_on_current_list'],
            row['Modest-Align_compatible_with_three_equal_50_image_top1_runs']))
    text += ['', '## 6. Surviving text-cache identity', '',
             '`cosine(a,b) = sum(a*b) / sqrt(sum(a*a)*sum(b*b))`.',
             '`allclose = every(abs(a-b) <= 2e-5 + 2e-5*abs(b))`.',
             'The 50 dictionary entries are 48 text vectors plus two maps, paired with three image vectors.',
             'The 48 vectors match three repetitions of separately processed characters, not whole prompts.',
             'Every comparison is in `text_fingerprint_calculations.csv`; this applies to this cache only.', '',
             '## 7. Limits', '',
             'The historical five Modest-Align percentages have NOT been reconstructed by this package.',
             'The new COCO evaluation and the single candidate-checkpoint evaluation are different regimes.',
             'The original classifier initializer, all historical group inputs and final aggregation remain unlinked.', '']
    (output / 'CALCULATIONS.md').write_text('\n'.join(text))


def check_manifest(root=ROOT):
    manifest = read_json(root / 'manifest.json')
    for name, expected in manifest['sha256'].items():
        path = root / name
        require(path.is_file(), 'Manifest file missing: ' + name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Hash mismatch: ' + name)
    return len(manifest['sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--check-manifest', action='store_true')
    args = parser.parse_args()
    if args.check_manifest:
        print('Manifest PASS:', check_manifest(), 'files')
    result = verify_and_recompute()
    write_report(result, args.output)
    print('PASS: {} image decisions; {} evaluation cells; {} legacy rows.'.format(
        result['records'], len(result['cells']), len(result['legacy_rows'])))
    print('Historical five-value reconstruction: NOT ESTABLISHED.')


if __name__ == '__main__':
    main()
