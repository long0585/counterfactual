import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import builtins
import argparse
import os

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--gpt_engine', type=str, default='gpt-4o', help='GPT-4 engine')
parser.add_argument('--alphabet', type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z', help='Choose custom alphabet')
parser.add_argument('--effort', type=str, help='Effort level for gpt5')
parser.add_argument('--composite', action='store_true', default=False)
args = parser.parse_args()

alphabet_suffix = args.alphabet.replace(" ", "")
effort_level = args.effort

# Load data
if not args.gpt_engine.startswith("gpt-5"):
    if args.composite:
        data_fname = './test_outputs_composite/' + args.gpt_engine + '_' + alphabet_suffix + '_int' + str(args.interval_size)
    else:
        data_fname = './test_outputs/' + args.gpt_engine + '_' + alphabet_suffix + '_int' + str(args.interval_size)
else:
    if args.composite:
        data_fname = './test_outputs_composite/' + args.gpt_engine + '_' + effort_level + '_' + alphabet_suffix + '_int' + str(args.interval_size)
    else:
        data_fname = './test_outputs/' + args.gpt_engine + '_' + effort_level + '_' + alphabet_suffix + '_int' + str(args.interval_size)
data_fname += '_results.npz'
all_responses = np.load(data_fname)['all_prob_type_responses']
N_prob_types = all_responses.shape[0]
N_trials_per_prob_type = all_responses.shape[1]
# Load problems
if args.interval_size == 1:
    if args.composite:
        all_prob = np.load('./all_prob_int1/all_prob_composite_' + alphabet_suffix + '_interval1.npz', allow_pickle=True)['all_prob']
    else:
        all_prob = np.load('./all_prob_int1/all_prob_' + alphabet_suffix + '_interval1.npz', allow_pickle=True)['all_prob']
elif args.interval_size == 2:
    if args.composite:
        all_prob = np.load('./all_prob_int2/all_prob_composite_' + alphabet_suffix + '_interval2.npz', allow_pickle=True)['all_prob']
    else:
        all_prob = np.load('./all_prob_int2/all_prob_' + alphabet_suffix + '_interval2.npz', allow_pickle=True)['all_prob']
prob_types = builtins.list(all_prob.item().keys())
# Avg time per problem (seconds)
avg_time = np.load(data_fname)['avg_time'].item()

# Calculate performance
all_prob_type_correct_pred = []
for p in range(N_prob_types):
    all_correct_pred = []
    print("-------------------", prob_types[p], "-------------------")
    for t in range(N_trials_per_prob_type):
        response = all_responses[p][t]
        response_parsed = response.split('Answer:')[-1].split('[')[-1].split(']')[0].split(' ')
        correct_answer = all_prob.item()[prob_types[p]]['prob'][t][1][1]
        if np.array(correct_answer).astype(str).shape[0] == np.array(response_parsed).astype(str).shape[0]:
            correct_pred = np.all(np.array(correct_answer).astype(str) == np.array(response_parsed))
        else:
            correct_pred = False
        if not correct_pred:
            print(all_prob.item()[prob_types[p]]['prob'][t][0][0], all_prob.item()[prob_types[p]]['prob'][t][0][1])
            print(all_prob.item()[prob_types[p]]['prob'][t][1][0])
            print(t, response_parsed, correct_answer)	# Print incorrect responses (to ensure there are no parsing errors)
            print()
        all_correct_pred.append(correct_pred)
    all_prob_type_correct_pred.append(all_correct_pred)
# Convert to arrays
all_prob_type_correct_pred = np.array(all_prob_type_correct_pred)

# Calculate accuracy for all problem types
if not args.composite:
    all_prob_types = ['add_letter', 'succ', 'pred', 'remove_redundant', 'fix_alphabet', 'sort']
else:
    all_prob_types = ['succ_pred', 'fix_alphabet_add_letter', 'sort_succ']
all_acc = []
all_ci_lower = []
all_ci_upper = []
ind_trial_results = []
for p in range(len(all_prob_types)):
    correct_pred = np.array(all_prob_type_correct_pred[np.where(np.array(prob_types)==all_prob_types[p])[0][0]]).astype(float)
    all_acc.append(correct_pred.mean())
    ci_lower, ci_upper = proportion_confint(correct_pred.sum(), correct_pred.shape[0])
    all_ci_lower.append(ci_lower)
    all_ci_upper.append(ci_upper)
    ind_trial_results.append(all_prob_type_correct_pred[np.where(np.array(prob_types)==all_prob_types[p])[0][0]])
all_acc = np.array(all_acc)
all_ci_lower = np.array(all_ci_lower)
all_ci_upper = np.array(all_ci_upper)
all_lower_err = all_acc - all_ci_lower
all_upper_err =  all_ci_upper - all_acc
all_err = np.array([all_lower_err, all_upper_err])
ind_trial_results = np.array(ind_trial_results)

# Create directory for results
if args.interval_size == 1:
    if args.composite:
        results_dir = './int1_results_composite/' + args.gpt_engine + '_' + alphabet_suffix + '_int' + str(args.interval_size)
    else:
        results_dir = './int1_results/' + args.gpt_engine + '_' + alphabet_suffix + '_int' + str(args.interval_size)
elif args.interval_size == 2:
    if args.composite:
        results_dir = './int2_results_composite/' + args.gpt_engine + '_' + alphabet_suffix + '_int' + str(args.interval_size)
    else:
        results_dir = './int2_results/' + args.gpt_engine + '_' + alphabet_suffix + '_int' + str(args.interval_size)
results_dir += '/'
check_path(results_dir)

# Plot settings
gpt_color = 'darkmagenta'
plot_fontsize = 10
title_fontsize = 12
axis_label_fontsize = 12
bar_width = 0.8

# Plot
if not args.composite:
    all_prob_type_names = ['Extend\nsequence', 'Successor', 'Predecessor', 'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
else:
    all_prob_type_names = ['Successor +\nPredecessor', 'Fix alphabetic\nsequence + Extend', 'Sort + Successor']
x_points = np.arange(len(all_prob_types))
ax = plt.subplot(111)
plt.bar(x_points, all_acc, yerr=all_err, color=gpt_color, edgecolor='black', width=bar_width)
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, np.array(all_prob_type_names), fontsize=plot_fontsize)
plt.xlabel('Problem type', fontsize=axis_label_fontsize)
plt.legend([args.gpt_engine],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(0.8, 1.02))

hide_top_right(ax)
plt.title("Alphabet:" + args.alphabet + '\n' + "Trials per problem type: " + str(N_trials_per_prob_type) + '\n' + "Interval size: " + str(args.interval_size) + '\n' + "Avg time/problem (s): " + str(round(avg_time, 2)))
plt.tight_layout()
if not args.gpt_engine.startswith('gpt-5'): 
    results_filename_png = "acc" + "_" + args.alphabet.replace(" ", "") + ".png"
else:
    results_filename_png = effort_level + "_" + args.alphabet.replace(" ", "") + ".png"
plt.savefig(results_dir + results_filename_png, dpi=300, bbox_inches="tight")
plt.close()
# Calculate overall accuracy
overall_acc = all_prob_type_correct_pred.mean()
ci_lower, ci_upper = proportion_confint(all_prob_type_correct_pred.sum(), all_prob_type_correct_pred.flatten().shape[0])
lower_err = overall_acc - ci_lower
upper_err =  ci_upper - overall_acc 
overall_err = np.array([lower_err, upper_err])
print('Overall accuracy = ' + str(overall_acc))
# Save results
if not args.gpt_engine.startswith("gpt-5"):
    results_filename_npz = "acc_" + "_" + args.alphabet.replace(" ", "") + ".npz"
else:
    results_filename_npz = effort_level + "_" + args.alphabet.replace(" ", "") + ".npz"
np.savez(results_dir + results_filename_npz, all_acc=all_acc, all_err=all_err, overall_acc=overall_acc, overall_err=overall_err, ind_trial_results=ind_trial_results, num_trials=N_trials_per_prob_type, avg_time=avg_time)
