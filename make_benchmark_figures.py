"""
Generate benchmarking figures comparing GPT-5.4, Qwen3.8-27B, Kimi-K3,
DeepSeek-V4-Pro, and GLM-5.3 on the counterfactual letter-string analogy task.

Produces:
  Figure 1: Interval-1 accuracy by problem type, all 5 models (grouped bars)
  Figure 2: Interval-2 accuracy by problem type, all 5 models (grouped bars)
  Figure 3: (a) overall accuracy leaderboard (int1 vs int2), (b) accuracy vs.
            average time-per-problem efficiency scatter (GPT-5.4 effort curve
            shown as a connected line; other models as single points)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import os

def safe_load(path, tries=6, delay=1.0):
    last_err = None
    for _ in range(tries):
        try:
            return np.load(path, allow_pickle=True)
        except OSError as e:
            last_err = e
            time.sleep(delay)
    raise last_err

def hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

ALPHABET = 'xylkwbfztnjrqahvgmuopdicse'
MODEL_DIRS = {
    'GPT-5.4': 'gpt-5.4',
    'Qwen3.8-27B': 'Qwen__Qwen3.8-27B__deepinfra_low',
    'Kimi-K3': 'moonshotai__Kimi-K3__deepinfra_low',
    'DeepSeek-V4-Pro': 'deepseek-ai__DeepSeek-V4-Pro__deepinfra_low',
    'GLM-5.3': 'zai-org__GLM-5.3__deepinfra_low',
}
MODEL_ORDER = ['GPT-5.4', 'Kimi-K3', 'DeepSeek-V4-Pro', 'Qwen3.8-27B', 'GLM-5.3']
COLORS = {
    'GPT-5.4': 'darkmagenta',
    'Kimi-K3': 'mediumseagreen',
    'DeepSeek-V4-Pro': 'royalblue',
    'Qwen3.8-27B': 'salmon',
    'GLM-5.3': 'goldenrod',
}
PROB_TYPE_NAMES = ['Extend\nsequence', 'Successor', 'Predecessor', 'Remove\nredundant\nletter',
                    'Fix\nalphabetic\nsequence', 'Sort']

def load_all():
    data = {1: {}, 2: {}}
    for interval in [1, 2]:
        for name, dirlabel in MODEL_DIRS.items():
            d = dirlabel + '_' + ALPHABET + '_int' + str(interval)
            path = f'int{interval}_results/{d}/'
            if dirlabel == 'gpt-5.4':
                data[interval][name] = {}
                for eff in ['high', 'medium', 'low']:
                    fp = path + eff + '_' + ALPHABET + '.npz'
                    npz = safe_load(fp)
                    data[interval][name][eff] = {k: npz[k] for k in npz.files}
            else:
                fp = path + 'acc_' + ALPHABET + '.npz'
                npz = safe_load(fp)
                data[interval][name] = {k: npz[k] for k in npz.files}
    return data

def primary(entry, name):
    """Return the representative single-condition record for a model
    (GPT-5.4 uses its high-effort configuration as the headline condition)."""
    if name == 'GPT-5.4':
        return entry['high']
    return entry

def make_problem_type_figure(data, interval, outpath):
    N_cond = 6
    x = np.arange(N_cond)
    n_models = len(MODEL_ORDER)
    total_width = 0.82
    bw = total_width / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bw

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, name in enumerate(MODEL_ORDER):
        rec = primary(data[interval][name], name)
        ax.bar(x + offsets[i], rec['all_acc'], yerr=rec['all_err'],
               width=bw * 0.92, color=COLORS[name], edgecolor='black',
               linewidth=0.6, ecolor='gray', capsize=2, label=name)
    ax.set_ylim([0, 1.08])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(PROB_TYPE_NAMES, fontsize=9.5)
    ax.set_xlabel('Transformation type', fontsize=13)
    n_trials = int(primary(data[interval]['GPT-5.4'], 'GPT-5.4')['num_trials'])
    ax.set_title(f'Interval size = {interval}\nTrials per problem type: {n_trials} per model (600 problems total)',
                 fontsize=13)
    ax.legend(fontsize=10, frameon=False, ncol=n_models, loc='upper center',
              bbox_to_anchor=(0.5, 1.30))
    hide_top_right(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

def make_summary_figure(data, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel A: overall accuracy leaderboard, int1 vs int2
    ax = axes[0]
    n_models = len(MODEL_ORDER)
    x = np.arange(n_models)
    bw = 0.36
    acc1 = [float(primary(data[1][m], m)['overall_acc']) for m in MODEL_ORDER]
    err1 = [primary(data[1][m], m)['overall_err'] for m in MODEL_ORDER]
    err1 = np.array(err1).T
    acc2 = [float(primary(data[2][m], m)['overall_acc']) for m in MODEL_ORDER]
    err2 = [primary(data[2][m], m)['overall_err'] for m in MODEL_ORDER]
    err2 = np.array(err2).T
    ax.bar(x - bw/2, acc1, yerr=err1, width=bw, color='powderblue',
           edgecolor='black', linewidth=0.6, ecolor='gray', capsize=2, label='Interval size 1')
    ax.bar(x + bw/2, acc2, yerr=err2, width=bw, color='indianred',
           edgecolor='black', linewidth=0.6, ecolor='gray', capsize=2, label='Interval size 2')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=9, rotation=20, ha='right')
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Overall accuracy', fontsize=12)
    ax.set_title('(a) Overall benchmark accuracy', fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    hide_top_right(ax)

    # Panel B: accuracy vs avg time per problem (efficiency), interval 1
    ax = axes[1]
    # GPT-5.4 effort curve
    effs = ['low', 'medium', 'high']
    gpt_times = [data[1]['GPT-5.4'][e]['avg_time'].item() for e in effs]
    gpt_accs = [data[1]['GPT-5.4'][e]['overall_acc'].item() for e in effs]
    ax.plot(gpt_times, gpt_accs, '-o', color=COLORS['GPT-5.4'], linewidth=1.5,
            markersize=7, label='GPT-5.4 (low→medium→high effort)')
    for i, e in enumerate(effs):
        ax.annotate(e, (gpt_times[i], gpt_accs[i]), textcoords="offset points",
                    xytext=(6, -3), fontsize=8)
    for name in MODEL_ORDER:
        if name == 'GPT-5.4':
            continue
        t = data[1][name]['avg_time'].item()
        a = data[1][name]['overall_acc'].item()
        ax.scatter([t], [a], color=COLORS[name], edgecolor='black', s=70, zorder=5, label=name)
    ax.set_xscale('log')
    ax.set_xlabel('Avg. time per problem (s, log scale)', fontsize=12)
    ax.set_ylabel('Overall accuracy', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_title('(b) Accuracy vs. inference cost (interval 1)', fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    hide_top_right(ax)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = load_all()
    make_problem_type_figure(data, 1, 'benchmark_fig1_int1_by_problem_type.png')
    make_problem_type_figure(data, 2, 'benchmark_fig2_int2_by_problem_type.png')
    make_summary_figure(data, 'benchmark_fig3_summary.png')

    # Print a results table for use in the paper text
    import json
    table = {}
    for interval in [1, 2]:
        table[interval] = {}
        for name in MODEL_ORDER:
            rec = primary(data[interval][name], name)
            table[interval][name] = {
                'overall_acc': float(rec['overall_acc']),
                'overall_err': rec['overall_err'].tolist(),
                'avg_time': float(rec['avg_time']),
                'all_acc': rec['all_acc'].tolist(),
            }
        table[interval]['GPT-5.4_medium'] = {
            'overall_acc': float(data[interval]['GPT-5.4']['medium']['overall_acc']),
            'avg_time': float(data[interval]['GPT-5.4']['medium']['avg_time']),
        }
        table[interval]['GPT-5.4_low'] = {
            'overall_acc': float(data[interval]['GPT-5.4']['low']['overall_acc']),
            'avg_time': float(data[interval]['GPT-5.4']['low']['avg_time']),
        }
    with open('benchmark_results_table.json', 'w') as f:
        json.dump(table, f, indent=2)
    print(json.dumps(table, indent=2))
    print("Figures saved.")
