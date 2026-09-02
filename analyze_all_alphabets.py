import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse interval size
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--gpt_engine', type=str, default="gpt-4o", help='gpt engine')
parser.add_argument('--effort', type=str)
args = parser.parse_args()

def hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
alphabet_map = {}    
with open(args.gpt_engine + "_alphabets.txt", "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if i % 2 == 0:
            alphabet_map[lines[i].strip()] = lines[i+1].strip().replace(" ", "")

# Load data
# newer GPT-4 engine
gpt_results = {}
for name, alph in alphabet_map.items():
    if args.interval_size == 1:
        if args.gpt_engine.startswith('gpt-5'):
            path = "./int1_results/" + args.gpt_engine + "_" + alph + "_int1/" + args.effort + "_" + alph + ".npz"
        else:
            path = "./int1_results/" + args.gpt_engine + "_" + alph + "_int1/" + "acc_" + alph + ".npz"
        gpt_results[name] = np.load(path)
    elif args.interval_size == 2:
        if args.gpt_engine.startswith("gpt-5"):
            path = "./int2_results/" + args.gpt_engine + "_" + alph + "_int2/" + args.effort + "_" + alph + ".npz"
        else:
            path = "./int2_results/" + args.gpt_engine + "_" + alph + "_int2/" + "acc_" + alph + ".npz"
        gpt_results[name] = np.load(path)
    N_trials_per_problem_type = gpt_results[name]['num_trials']
# gpt4_int2_results = np.load('./gpt-4-0125-preview_int2/acc.npz')
# older GPT-4 engine
# old_gpt4_int1_results = np.load('./gpt-4-1106-preview_int1/acc.npz')
# old_gpt4_int2_results = np.load('./gpt-4-1106-preview_int2/acc.npz')

# Get accuracy for each condition
# newer GPT-4 engine
# Interval size = 1
gpt_acc = [result['overall_acc'].item() for result in gpt_results.values()]
gpt_err = [result['overall_err'][0] for result in gpt_results.values()]
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
ind_bar_width = total_bar_width / 2
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
plt.bar(x_points + (ind_bar_width * 1/2), gpt_results["FORWARD"]['all_acc'], yerr=gpt_results["FORWARD"]['all_err'], color=colors[0], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points - (ind_bar_width * 1/2), gpt_results["RANDOM"]['all_acc'], yerr=gpt_results["RANDOM"]['all_err'], color=colors[1], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Problem type', fontsize=axis_label_fontsize)
if args.gpt_engine == 'gpt-4o':
    plt.title('GPT model: ' + args.gpt_engine + '\nInterval size = ' + str(args.interval_size) + '\n' + "Trials per problem type: " + str(N_trials_per_problem_type), fontsize=title_fontsize)
elif args.gpt_engine.startswith('gpt-5'):
    plt.title('GPT model: ' + args.gpt_engine + '\nInterval size = ' + str(args.interval_size) + '\n' + "Effort: " + args.effort + '\n' + "Trials per problem type: " + str(N_trials_per_problem_type), fontsize=title_fontsize)
plt.legend([name.lower() for name in gpt_results.keys()],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(1.1, 1))
hide_top_right(ax)
if args.interval_size == 1:
    if args.gpt_engine.startswith('gpt-5'):
        plt.savefig('./' + args.gpt_engine + '_int1_' + args.effort + '_combined_results.png', dpi=300, bbox_inches="tight")
    elif args.gpt_engine == 'gpt-4o':
        plt.savefig('./gpt4o_int1_combined_results.png', dpi=300, bbox_inches="tight")
elif args.interval_size == 2:
    if args.gpt_engine.startswith('gpt-5'):
        plt.savefig('./' + args.gpt_engine + '_int2_' + args.effort + '_combined_results.png', dpi=300, bbox_inches="tight")
    elif args.gpt_engine == 'gpt-4o':
        plt.savefig('./gpt4o_int2_combined_results.png', dpi=300, bbox_inches="tight")
plt.close()