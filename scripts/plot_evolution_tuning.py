"""Render the saved-history diagnostics; no model execution or feature loading."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path.cwd()
audit = json.loads((root / 'results/evolution_tuning_audit.json').read_text())
writers = ['Qwen/Qwen3.8-27B', 'poolside/Laguna-XS-2.1', 'Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-4B']
labels = ['Qwen27B', 'Laguna XS', 'Qwen9B', 'Qwen4B']
valid = [100*audit['by_writer'][w]['valid_fraction'] for w in writers]
repeat = [100*audit['by_writer'][w]['rank_near_repeat_fraction'] for w in writers]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
for ax, values, title, color in zip(axes, [valid, repeat],
        ['Executable candidates / requests', 'Near repeats / executable candidates'], ['#277a91', '#ae7149']):
    ax.barh(labels, values, color=color, height=.6)
    for i, value in enumerate(values):
        ax.text(value+1.5, i, f'{value:.1f}%', va='center', fontsize=10)
    ax.set_xlim(0, 106)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Percent')
    ax.set_title(title, fontsize=11, pad=12)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
axes[0].invert_yaxis()
fig.suptitle('EvoHallu: search reliability and prediction diversity', fontsize=13)
fig.text(.5, .025,
    'Seven recorded evolution runs. Near repeat: saved validation-rank Spearman ≥ 0.995 to baseline/earlier candidate.\n'
    'Posthoc diagnostic, not an AUROC gain. Constant predictions are excluded from correlation matches.',
    ha='center', fontsize=8, color='#45505c')
fig.tight_layout(rect=[0, .13, 1, .95], w_pad=2)
destination = root / 'results/figures'
destination.mkdir(exist_ok=True)
for ext in ['png', 'pdf']:
    fig.savefig(destination / f'evolution_tuning_diagnostics.{ext}', dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved results/figures/evolution_tuning_diagnostics.png and .pdf')
