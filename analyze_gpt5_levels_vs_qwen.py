import numpy as np
import matplotlib.pyplot as plt
import argparse

def hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

parser = argparse.ArgumentParser()
parser.add_argument('--alphabet', type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z', help='Choose custom alphabet')
parser.add_argument('--interval_size', type=int, default=1, help='interval size')
parser.add_argument('--gpt_engine', type=str, default='gpt-5.4', help='effort-level model to compare across high/medium/low')
# NOTE: --compare_label must be the *sanitized* result_model_label used for the
# comparison model's results directory (e.g. "Qwen__Qwen3.8-27B__deepinfra_low"),
# not the raw --gpt_engine value passed to test_gpt.py/analyze_gpt.py. See the
# comment in analyze_all_alphabets.py for the same gotcha.
parser.add_argument('--compare_label', type=str, default='Qwen__Qwen3.8-27B__deepinfra_low', help='Sanitized result_model_label for the non-effort-level comparison model')
parser.add_argument('--compare_name', type=str, default='Qwen', help='Display name for the comparison model in the legend/title')
args = parser.parse_args()

alphabet = args.alphabet.replace(" ", "")
effort_levels = ["high", "medium", "low"]

gpt5_results = {}
for effort_level in effort_levels:
    if args.interval_size == 1:
        path = "./int1_results/" + args.gpt_engine + '_' + alphabet + "_int1/" + effort_level + "_" + alphabet + ".npz"
    elif args.interval_size == 2:
        path = "./int2_results/" + args.gpt_engine + '_' + alphabet + "_int2/" + effort_level + "_" + alphabet + ".npz"
    gpt5_results[effort_level] = np.load(path)

# Load the comparison model's single result (no effort levels -- e.g. Qwen),
# saved by analyze_gpt.py under the "acc_<alphabet>.npz" filename.
if args.interval_size == 1:
    compare_path = "./int1_results/" + args.compare_label + '_' + alphabet + "_int1/" + "acc_" + alphabet + ".npz"
elif args.interval_size == 2:
    compare_path = "./int2_results/" + args.compare_label + '_' + alphabet + "_int2/" + "acc_" + alphabet + ".npz"
compare_results = np.load(compare_path)

# Grab number of trials from arbitrary entry
N_trials_per_problem_type = (gpt5_results["low"]['num_trials'], gpt5_results["medium"]['num_trials'], gpt5_results["high"]['num_trials'])
compare_N_trials_per_problem_type = compare_results['num_trials']

# Get accuracy for each condition
gpt5_int1_acc = [result['overall_acc'].item() for result in gpt5_results.values()]
gpt5_int1_err = [result['overall_err'][0] for result in gpt5_results.values()]

# Plot parameters
total_bar_width = 0.8
ind_bar_width = total_bar_width / 4
colors = ['powderblue', 'darkmagenta', 'salmon', 'mediumseagreen', 'royalblue']
plot_fontsize = 14
title_fontsize = 16
axis_label_fontsize = 14
## Plot separately for different interval-size conditions
all_prob_type_names = ['Extend\nsequence', 'Successor', 'Predecessor', 'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
N_cond = 6
x_points = np.arange(N_cond)
# Interval size = 1
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width * 1.5), gpt5_results["high"]['all_acc'], yerr=gpt5_results["high"]['all_err'], color=colors[0], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points - (ind_bar_width * 0.5), gpt5_results["medium"]['all_acc'], yerr=gpt5_results["medium"]['all_err'], color=colors[1], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width * 0.5), gpt5_results["low"]['all_acc'], yerr=gpt5_results["low"]['all_err'], color=colors[2], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width * 1.5), compare_results['all_acc'], yerr=compare_results['all_err'], color=colors[3], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.title(args.gpt_engine + ' vs ' + args.compare_name + '\nAlphabet: ' + args.alphabet + '\n' + 'Interval size = ' + str(args.interval_size) + '\n' + "Trials per problem type: " + str(N_trials_per_problem_type[2]) + " (" + args.compare_name + ": " + str(compare_N_trials_per_problem_type) + ")", fontsize=title_fontsize)
plt.legend([name.lower() for name in gpt5_results.keys()] + [args.compare_name.lower()], fontsize=plot_fontsize, frameon=False, bbox_to_anchor=(1.1, 1))
hide_top_right(ax)
if args.interval_size == 1:
    plt.savefig('./' + args.gpt_engine + '_vs_' + args.compare_name + '_int1_' + alphabet + '_combined_results.png', dpi=300, bbox_inches="tight")
elif args.interval_size == 2:
    plt.savefig('./' + args.gpt_engine + '_vs_' + args.compare_name + '_int2_' + alphabet + '_combined_results.png', dpi=300, bbox_inches="tight")
plt.close()
