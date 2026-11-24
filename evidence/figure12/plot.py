"""Generate scientific figures from independently recomputed values."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from recompute import DATASETS, verify_and_recompute

ROOT = Path(__file__).resolve().parent
COLORS = {'CLIP': '#597080', 'COCO_RP_ES': '#c74444', 'candidate_20240906195': '#23816a'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.titlepad': 12})


def save(fig, name):
    fig.savefig(ROOT / (name + '.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(ROOT / (name + '.pdf'), bbox_inches='tight', facecolor='white',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    result = verify_and_recompute()
    cells = result['cells']
    y = np.arange(len(DATASETS))
    with (ROOT / 'historical_reported_values.csv').open(newline='') as stream:
        historical = {r['dataset']: r for r in csv.DictReader(stream)}
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.subplots_adjust(left=.17, right=.98, bottom=.13, top=.76)
    for field, offset, color, label in [('CLIP', -.19, COLORS['CLIP'], 'Historical reported CLIP'),
                                       ('Modest-Align', .19, COLORS['COCO_RP_ES'], 'Historical reported Modest-Align')]:
        values = [float(historical[n][field]) for n in DATASETS]
        ax.barh(y + offset, values, .35, color=color, label=label)
        for i, value in enumerate(values):
            ax.text(value + 1, i + offset, '{:.2f}'.format(value), va='center', fontsize=9)
    ax.set_yticks(y, [n.replace('BrainTumor', 'Brain tumor') for n in DATASETS])
    ax.invert_yaxis()
    ax.set_xlim(0, 114)
    ax.set_xlabel('Historical reported percentage (metric aggregation unverified)')
    fig.suptitle('Historical Figure-12 values - NOT reconstructed', fontsize=13, y=.995)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.55, .92), ncol=2, frameon=False)
    ax.grid(axis='x', alpha=.16)
    ax.set_axisbelow(True)
    save(fig, 'historical_reported')

    labels = [n.replace('BrainTumor', 'Brain tumor') + '\n(n={})'.format(
        next(c['n'] for c in cells if c['dataset'] == n)) for n in DATASETS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), sharey=True)
    fig.subplots_adjust(left=.11, right=.985, bottom=.12, top=.78, wspace=.12)
    for ax, metric, title in zip(axes, ['accuracy_percent', 'balanced_accuracy_percent'],
                                ['Top-1 accuracy', 'Balanced accuracy']):
        baseline = [next(c[metric] for c in cells if c['dataset'] == n and c['method'] == 'CLIP') for n in DATASETS]
        agg = [next(a for a in result['coco_aggregates'] if a['dataset'] == n and a['metric'] == metric) for n in DATASETS]
        means, sds = [a['mean'] for a in agg], [a['sample_sd'] for a in agg]
        ax.barh(y - .19, baseline, .35, color=COLORS['CLIP'], label='CLIP ViT-B/32')
        ax.barh(y + .19, means, .35, xerr=sds, capsize=3, color=COLORS['COCO_RP_ES'], label='COCO RP+ES: all 5 adapters')
        for i, (base, mean, sd) in enumerate(zip(baseline, means, sds)):
            ax.text(base + 1, i - .19, '{:.2f}'.format(base), va='center', fontsize=9)
            ax.text(mean + sd + 1, i + .19, '{:.2f}'.format(mean), va='center', fontsize=9)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Percent')
        ax.set_title(title, loc='left')
        ax.grid(axis='x', alpha=.16)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.55, .93), ncol=2, frameon=False)
    fig.suptitle('New class-level evaluation: COCO-trained adapters', fontsize=13, y=.995)
    save(fig, 'new_classification')

    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    fig.subplots_adjust(left=.18, right=.985, bottom=.13, top=.77)
    for method, offset, label in [('CLIP', -.19, 'CLIP ViT-B/32'),
                                  ('candidate_20240906195', .19, 'Candidate 20240906195 (one checkpoint)')]:
        vals = [next(c['accuracy_percent'] for c in cells if c['dataset'] == n and c['method'] == method) for n in DATASETS]
        ax.barh(y + offset, vals, .35, color=COLORS[method], label=label)
        for i, value in enumerate(vals):
            ax.text(value + 1, i + offset, '{:.2f}'.format(value), va='center', fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xlabel('Top-1 accuracy (%)')
    fig.suptitle('Reconstructed classifier setup; not a historical Figure-12 reproduction', fontsize=11, y=.995)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.57, .94), frameon=False, fontsize=9)
    ax.grid(axis='x', alpha=.16)
    ax.set_axisbelow(True)
    save(fig, 'candidate_checkpoint')

    chosen = [next(r for r in result['legacy_rows'] if r['run_id'] == run and r['line'] == str(line))
              for run, line in [('20241121113', 1), ('20241121113', 2), ('20241121114', 1)]]
    fig, ax = plt.subplots(figsize=(7, 3.8), layout='constrained')
    x = np.arange(len(chosen))
    ax.bar(x - .19, [r['txt_r1'] for r in chosen], .35, color=COLORS['CLIP'], label='Logged txt_r1')
    ax.bar(x + .19, [r['agg_metrics'] for r in chosen], .35, color='#b97924', label='Logged mean of R@1 / R@5 / R@10')
    for i, row in enumerate(chosen):
        for offset, key in [(-.19, 'txt_r1'), (.19, 'agg_metrics')]:
            ax.text(i + offset, row[key] + 2, '{:.2f}'.format(row[key]), ha='center', fontsize=10)
    ax.set_xticks(x, [r['run_id'] + '\nline ' + r['line'] for r in chosen])
    ax.set_ylim(0, 115)
    ax.set_ylabel('Recorded percent')
    ax.set_title('Different recorded statistics from the same evaluation rows', fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    save(fig, 'legacy_metric_definitions')

    fingerprint = result['text_fingerprint_rows']
    fig, ax = plt.subplots(figsize=(7, 3.6), layout='constrained')
    positions = list(range(16))
    ax.plot(positions, [r['character_cosine'] for r in fingerprint[:16]], 'o-', color='#23816a', label='Cached vector vs. separately encoded character')
    ax.plot(positions, [r['whole_caption_cosine'] for r in fingerprint[:16]], 's-', color=COLORS['COCO_RP_ES'], label='Cached vector vs. whole-caption vector')
    ax.set_ylim(.45, 1.06)
    ax.set_xticks(positions)
    ax.set_xlabel('Text-cache position in one 16-vector block')
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Identity check of one surviving text cache', fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc='center right')
    ax.grid(alpha=.16)
    save(fig, 'text_cache_identity')
    print('Generated 5 PNG/PDF pairs: four computed diagnostics and one clearly labeled historical display.')


if __name__ == '__main__':
    main()
