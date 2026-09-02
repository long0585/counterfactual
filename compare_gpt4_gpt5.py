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
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
args = parser.parse_args()
# Load data
effort_levels = ["high", "medium", "low"]
alphabet = args.alphabet.replace(" ", "")
gpt_results = {}
# Add gpt5 results
for effort_level in effort_levels:
    if args.interval_size == 1:
        path = "./int1_results/gpt-5.4_" + alphabet + "_int1/" + effort_level + "_" + alphabet + ".npz"
    elif args.interval_size == 2:
        path = "./int2_results/gpt-5.4_" + alphabet + "_int2/" + effort_level + "_" + alphabet + ".npz"
    gpt_results[effort_level] = np.load(path)
# Add human results
if alphabet == "xylkwbfztnjrqahvgmuopdicse":
    if args.interval_size == 1:
        gpt_results["human"] = np.load('./behavioral_data_int1/acc.npz')
    elif args.interval_size == 2:
        gpt_results["human"] = np.load('./behavioral_data_int2/acc.npz')
# Add gpt4 results
if args.interval_size == 1:
    gpt_results["gpt4o"] = np.load("./int1_results/gpt-4o_" + alphabet + "_int1/acc_" + alphabet + ".npz")
elif args.interval_size == 2:
    gpt_results["gpt4o"] = np.load("./int2_results/gpt-4o_" + alphabet + "_int2/acc_" + alphabet + ".npz")


# gpt4_int2_results = np.load('./gpt-4-0125-preview_int2/acc.npz')
# older GPT-4 engine
# old_gpt4_int1_results = np.load('./gpt-4-1106-preview_int1/acc.npz')
# old_gpt4_int2_results = np.load('./gpt-4-1106-preview_int2/acc.npz')

# Get accuracy for each condition
# newer GPT-4 engine
# "int1" isn't accurate in this context as a variable name
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
ind_bar_width = total_bar_width / 5
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
plt.bar(x_points - (ind_bar_width * 2), gpt_results["high"]['all_acc'], yerr=gpt_results["high"]['all_err'], color=colors[0], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points - (ind_bar_width * 1), gpt_results["medium"]['all_acc'], yerr=gpt_results["medium"]['all_err'], color=colors[1], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width * 0), gpt_results["low"]['all_acc'], yerr=gpt_results["low"]['all_err'], color=colors[2], edgecolor='black', width=ind_bar_width, ecolor='gray')
if alphabet == "xylkwbfztnjrqahvgmuopdicse":
    plt.bar(x_points + (ind_bar_width * 1), gpt_results["human"]['all_acc'], yerr=gpt_results["human"]['all_err'], color=colors[3], edgecolor='black', width=ind_bar_width, ecolor='black')
    plt.bar(x_points + (ind_bar_width * 2), gpt_results["gpt4o"]['all_acc'], yerr=gpt_results["gpt4o"]['all_err'], color=colors[4], edgecolor='black', width=ind_bar_width, ecolor='gray')
else:
    plt.bar(x_points + (ind_bar_width * 1), gpt_results["gpt4o"]['all_acc'], yerr=gpt_results["gpt4o"]['all_err'], color=colors[3], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Problem type', fontsize=axis_label_fontsize)
plt.title('GPT-5.4 vs. GPT-4o vs. human results\n'+ 'Interval size = ' + str(args.interval_size) + '\n' + "Alphabet: " + alphabet, fontsize=title_fontsize)
plt.legend([name.lower() for name in gpt_results.keys()],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(1.1, 1))
hide_top_right(ax)
if args.interval_size == 1:
    plt.savefig('./gpt4o_gpt5.4_int1_' + alphabet + '_combined_results.png', dpi=300, bbox_inches="tight")
elif args.interval_size == 2:
    plt.savefig('./gpt4o_gpt5.4_int2_' + alphabet + '_combined_results.png', dpi=300, bbox_inches="tight")
plt.close()