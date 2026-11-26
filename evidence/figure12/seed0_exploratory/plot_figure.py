"""Exploratory seed-0 line plot from archived, independently checked predictions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(root / 'predictions.jsonl')]
names = ['CheXpert', 'Shenzhen', 'BrainTumor', 'IDRiD', 'D&C']
labels = ['CheXpert', 'Shenzhen', 'Brain Tumor', 'IDRiD', 'D&C']
def acc(d, m, s):
    rs = [r for r in rows if r['dataset'] == d and r['method'] == m and r['seed'] == s]
    return 100 * np.mean([r['prediction'] == r['target'] for r in rs])
clip = [acc(n, 'CLIP', None) for n in names]
ma = [acc(n, 'Modest-Align', 0) for n in names]
json.dump({'CLIP': dict(zip(names, clip)), 'Modest-Align_seed0': dict(zip(names, ma))}, open(root / 'figure_values.json', 'w'), indent=1)
plt.rcParams.update({'font.family': 'Times New Roman', 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(5.6, 3.5))
x = np.arange(len(names))
cb, cg = '#3a75b0', '#4f9a3c'
ax.plot(x, clip, '-o', color=cb, lw=1.8, ms=6, label='CLIP')
ax.plot(x, ma, '-d', color=cg, lw=1.8, ms=7, label='Modest-Align')
for i, (c, m) in enumerate(zip(clip, ma)):
    up_clip = c >= m
    dc, dm = ((0, 10), (0, -17)) if up_clip else ((0, -17), (0, 10))
    hc = 'center'
    if i == len(names) - 1:
        dc, dm = (0, 10), (10, -16)
    hm = 'left' if i == len(names) - 1 else 'center'
    if names[i] == 'Shenzhen':
        dm, hm = (8, -16), 'left'
    if names[i] == 'IDRiD':
        dc, hc = (9, -4), 'left'
    ax.annotate(f'{c:.2f}', (i, c), textcoords='offset points', xytext=dc, ha=hc, fontsize=8.5, color=cb)
    ax.annotate(f'{m:.2f}', (i, m), textcoords='offset points', xytext=dm, ha=hm, fontsize=8.5, color=cg)
ax.set_xlim(-0.4, 4.6)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(45, 107)
ax.set_title('Performance Comparison on Multiple Datasets', fontsize=10.5)
ax.grid(True, ls='--', alpha=0.4); ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(root / 'figure_seed0.pdf'); fig.savefig(root / 'figure_seed0.png', dpi=200)
