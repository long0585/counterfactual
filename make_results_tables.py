"""
Render Figure 1 and Figure 2 data (interval-1 and interval-2 accuracy by
problem type, all 5 models) as neatly formatted table images.

Reads benchmark_results_table.json (produced by make_benchmark_figures.py)
and writes:
  benchmark_table1_int1.png
  benchmark_table2_int2.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt

MODEL_ORDER = ['GPT-5.4', 'Kimi-K3', 'DeepSeek-V4-Pro', 'Qwen3.8-27B', 'GLM-5.3']
COLORS = {
    'GPT-5.4': 'darkmagenta',
    'Kimi-K3': 'mediumseagreen',
    'DeepSeek-V4-Pro': 'royalblue',
    'Qwen3.8-27B': 'salmon',
    'GLM-5.3': 'goldenrod',
}
PROB_TYPE_NAMES = ['Extend\nsequence', 'Successor', 'Predecessor',
                    'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
PROB_TYPE_HEADERS = ['Extend\nseq.', 'Succ.', 'Pred.', 'Remove\nredund.',
                      'Fix\nalpha.', 'Sort', 'Overall']


def make_table(data, interval, outpath):
    rows = []
    for name in MODEL_ORDER:
        rec = data[interval][name]
        vals = rec['all_acc']
        overall = rec['overall_acc']
        rows.append(vals + [overall])

    rows = np.array(rows)  # (n_models, 7)
    n_models = len(MODEL_ORDER)
    n_cols = rows.shape[1]

    fig_w = 1.15 * n_cols + 2.0
    fig_h = 0.62 * n_models + 1.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    col_labels = PROB_TYPE_HEADERS
    row_labels = MODEL_ORDER

    cell_text = [[f'{v*100:.0f}%' for v in row] for row in rows]

    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                      cellLoc='center', rowLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 2.05)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#2b2b2b')
        cell.set_text_props(color='white', weight='bold')
        cell.set_height(cell.get_height() * 1.3)

    # Style row labels (model names) with their chart colors
    for i, name in enumerate(row_labels):
        cell = table[i + 1, -1]  # placeholder to force creation order not needed
    for i, name in enumerate(row_labels):
        rl = table.get_celld()[(i + 1, -1)]
        rl.set_facecolor(COLORS[name])
        rl.set_text_props(color='white', weight='bold')

    # Shade overall-accuracy column and alternate row stripes
    for i in range(n_models):
        for j in range(n_cols):
            cell = table[i + 1, j]
            if j == n_cols - 1:
                cell.set_facecolor('#dde6f0')
                cell.set_text_props(weight='bold')
            elif i % 2 == 1:
                cell.set_facecolor('#f4f4f4')

    ax.set_title(f'Interval size = {interval}: accuracy by problem type (100 trials/type/model)',
                 fontsize=12.5, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    with open('benchmark_results_table.json') as f:
        data = json.load(f)
    # json keys are strings
    data = {1: data['1'], 2: data['2']}
    make_table(data, 1, 'benchmark_table1_int1.png')
    make_table(data, 2, 'benchmark_table2_int2.png')
    print('Tables saved.')
