import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse interval size
parser = argparse.ArgumentParser()
parser.add_argument('--gpt_engine', type=str, default="gpt-5.4", help='gpt engine')
args = parser.parse_args()
effort_levels = ['low', 'medium', 'high']
interval_sizes = [1, 2]

def hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
alphabet_map = {}    
with open(args.gpt_engine + "_alphabets.txt", "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if i % 2 == 0 and i != 0: # Hardcoded to only include random alphabet
            alphabet_map[lines[i].strip()] = lines[i+1].strip().replace(" ", "")


# Load data
# newer GPT-4 engine
gpt_results = {}
for name, alph in alphabet_map.items():
    for effort in effort_levels:
        for interval in interval_sizes:
            key = effort + '_' + name + '_' + str(interval)
            path = "./int" + str(interval) + "_results/" + args.gpt_engine + "_" + alph + "_int" + str(interval) + "/" + effort + "_" + alph + ".npz"
            gpt_results[key] = np.load(path)
            N_trials_per_problem_type = gpt_results[key]['num_trials']
            if interval == 1:
                key = effort + '_' + name + '_composite'
                path = "./int" + str(interval) + "_results_composite/" + args.gpt_engine + "_" + alph + "_int" + str(interval) + "/" + effort + "_" + alph + ".npz"
                gpt_results[key] = np.load(path)
                N_trials_per_problem_type = gpt_results[key]['num_trials']
# Include composite results

# Get time per effort and problem
low_times_1, med_times_1, high_times_1 = [], [], []
low_times_2, med_times_2, high_times_2 = [], [], []
low_times_composite, med_times_composite, high_times_composite = [], [], []
for key, val in gpt_results.items():
    time = val['avg_time'].item()
    if key.startswith('low'):
        if key.endswith('1'):
            low_times_1.append(time)
        elif key.endswith('2'):
            low_times_2.append(time)
        elif key.endswith('composite'):
            low_times_composite.append(time)
    elif key.startswith('medium'):
        if key.endswith('1'):
            med_times_1.append(time)
        elif key.endswith('2'):
            med_times_2.append(time)
        elif key.endswith('composite'):
            med_times_composite.append(time)
    elif key.startswith('high'):
        if key.endswith('1'):
            high_times_1.append(time)
        elif key.endswith('2'):
            high_times_2.append(time)
        elif key.endswith('composite'):
            high_times_composite.append(time)
    else:
        raise Exception ("Effort level is not low, medium, or high")
low_avg_1, med_avg_1, high_avg_1 = np.mean(low_times_1), np.mean(med_times_1), np.mean(high_times_1)
low_avg_2, med_avg_2, high_avg_2 = np.mean(low_times_2), np.mean(med_times_2), np.mean(high_times_2)
low_avg_composite, med_avg_composite, high_avg_composite = np.mean(low_times_composite), np.mean(med_times_composite), np.mean(high_times_composite)
labels = ['Low effort', 'Medium effort', 'High effort']
x = np.arange(len(labels))
values_1 = np.array([low_avg_1, med_avg_1, high_avg_1])
values_2 = np.array([low_avg_2, med_avg_2, high_avg_2])
values_3 = np.array([low_avg_composite, med_avg_composite, high_avg_composite])
ind_bar_width = 0.3
colors = ['powderblue', 'darkmagenta', 'salmon']
plot_fontsize = 14
title_fontsize = 16
axis_label_fontsize = 14
bars1 = plt.bar(x - ind_bar_width, values_1, ind_bar_width, label='Interval 1', color=colors[0], edgecolor='black')
bars2 = plt.bar(x, values_2, ind_bar_width, label='Interval 2', color=colors[1], edgecolor='black')
bars3 = plt.bar(x + ind_bar_width, values_3, ind_bar_width, label='Composite', color=colors[2], edgecolor='black')
plt.ylabel('Average Time (s)')
plt.title('Average Time Per Problem by Effort Level (s)')
plt.xticks(x, labels)
plt.legend()
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom')
plt.tight_layout()
plt.savefig('./' + args.gpt_engine + '_time_results.png', dpi=300, bbox_inches="tight")