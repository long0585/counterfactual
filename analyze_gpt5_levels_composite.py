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
parser.add_argument('--gpt_engine', type=str, default='gpt-5.4')
args = parser.parse_args()

alphabet = args.alphabet.replace(" ", "")
effort_levels = ["high", "medium", "low"]

gpt5_results = {}
for effort_level in effort_levels:
    if args.interval_size == 1:
        path = "./int1_results_composite/" + args.gpt_engine + '_' + alphabet + "_int1/" + effort_level + "_" + alphabet + ".npz"
    elif args.interval_size == 2:
        path = "./int2_results_composite/" + args.gpt_engine + '_' + alphabet + "_int2/" + effort_level + "_" + alphabet + ".npz"
    gpt5_results[effort_level] = np.load(path)
# Grab number of trials from arbitrary entry
N_trials_per_problem_type = (gpt5_results["low"]['num_trials'], gpt5_results["medium"]['num_trials'], gpt5_results["high"]['num_trials'])
# gpt4_int2_results = np.load('./gpt-4-0125-preview_int2/acc.npz')
# older GPT-4 engine
# old_gpt4_int1_results = np.load('./gpt-4-1106-preview_int1/acc.npz')
# old_gpt4_int2_results = np.load('./gpt-4-1106-preview_int2/acc.npz')

# Get accuracy for each condition
# newer GPT-4 engine
# Interval size = 1
gpt5_int1_acc = [result['overall_acc'].item() for result in gpt5_results.values()]
gpt5_int1_err = [result['overall_err'][0] for result in gpt5_results.values()]
# Interval size = 2
# gpt4_int2_acc = gpt4_int2_results['overall_acc'].item()
# gpt4_int2_err = gpt4_int1_results['overall_err'][0]
# Combined
# gpt4_acc = np.array([gpt4_int1_acc, gpt4_int2_acc])
# gpt4_err = np.array([gpt4_int1_err, gpt4_int2_err])
# older GPT-4 engine
# Interval size = 1
# old_gpt4_int1_acc = old_gpt4_int1_results['overall_acc'].item()
# old_gpt4_int1_err = old_gpt4_int1_results['overall_err'][0]
# Interval size = 2
# old_gpt4_int2_acc = old_gpt4_int2_results['overall_acc'].item()
# old_gpt4_int2_err = old_gpt4_int1_results['overall_err'][0]
# Combined
# old_gpt4_acc = np.array([old_gpt4_int1_acc, old_gpt4_int2_acc])
# old_gpt4_err = np.array([old_gpt4_int1_err, old_gpt4_int2_err])

# Plot parameters
total_bar_width = 0.8
ind_bar_width = total_bar_width / 4
colors = ['powderblue', 'darkmagenta', 'salmon', 'mediumseagreen', 'royalblue']
plot_fontsize = 14
title_fontsize = 16
axis_label_fontsize = 14
## Plot separately for different interval-size conditions
all_prob_type_names = ['Successor +\nPredecessor', 'Fix alphabetic\nsequence + Extend', 'Sort + Successor']
N_cond = 3
x_points = np.arange(N_cond)
# Interval size = 1
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width * 1.5), gpt5_results["high"]['all_acc'], yerr=gpt5_results["high"]['all_err'], color=colors[0], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points - (ind_bar_width * 0.5), gpt5_results["medium"]['all_acc'], yerr=gpt5_results["medium"]['all_err'], color=colors[1], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width * 0.5), gpt5_results["low"]['all_acc'], yerr=gpt5_results["low"]['all_err'], color=colors[2], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Problem type', fontsize=axis_label_fontsize)
plt.title('GPT-5.4 results for composite problem types\n Alphabet: ' + args.alphabet + '\n' + 'Interval size = ' + str(args.interval_size) + '\n' + "Trials per problem type: " + str(N_trials_per_problem_type[2]), fontsize=title_fontsize)
plt.legend([name.lower() for name in gpt5_results.keys()],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(1.1, 1))
hide_top_right(ax)
if args.interval_size == 1:
    plt.savefig('./' + args.gpt_engine + 'composite_int1_' + alphabet + '_combined_results.png', dpi=300, bbox_inches="tight")
elif args.interval_size ==2:
    plt.savefig('./' + args.gpt_engine + 'composite_int2_' + alphabet + '_combined_results.png', dpi=300, bbox_inches="tight")
plt.close()