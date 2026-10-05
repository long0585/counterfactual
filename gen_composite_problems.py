import os
import shutil

import numpy as np
from copy import deepcopy
import random
import json

import pandas as pd
from get_arguments import get_suffix_modified, args, get_version_dir

## Problems generated using one of the following 6 transformations:
#
# Successorship
# [a b c d] [a b c e]
# [i j k l] [i j k m]
#
# Predecessorship
# [b c d e] [a c d e]
# [j k l m] [i k l m]
#
# Add letter to sequence
# [a b c d] [a b c d e]
# [i j k l] [i j k l m]
#
# Remove redundant character
# [a b b c d e] [a b c d e]
# [i j k l l m] [i j k l m]
#
# Fix alphabetic sequence
# [a w c d e] [a b c d e]
# [i j k p m] [i j k l m]
#
# Sort characters
# [a b e d c] [a b c d e]
# [i k j l m] [i j k l m]
#
#
## and between 0 and 3 generalizations, out of the following 6:
#
# Larger interval
# [a b c d] [a b c e]
# [i k m o] [i k m q]
#
# Longer target
# [a b c d] [a b c e]
# [i j k l m n o p] [i j k l m n o q]
#
# Grouping
# [a b c d] [a b c e]
# [i i j j k k l l] [i i j j k k m m]
#
# Interleaved X's
# [a b c d] [a b c e]
# [i x j x k x l x] [i x j x k x m x] 
#
# Letter-to-number
# [a b c d] [a b c e]
# [1 2 3 4] [1 2 3 5]
#
# Reversal
# [a b c d] [a b c e]
# [m l k j] [m l k i]
#
##
if args.modified:
    INTERVAL_MOD = 1  # added to the interval of 1 used in the paper
    INTERVAL_SIZE =  INTERVAL_MOD + 1 # in paper = 2
else:
    INTERVAL_MOD = 0
    INTERVAL_SIZE =  INTERVAL_MOD + 1

# Alphabet
if args.synthetic:
    # random.seed(1)
    # random.shuffle(letters,)
    if args.customalphabet is not None:
        letters = args.customalphabet.split()
    else:
        letters = ['x', 'y', 'l', 'k', 'w', 'b', 'f', 'z', 't', 'n', 'j', 'r', 'q', 'a', 'h', 'v', 'g', 'm', 'u', 'o', 'p', 'd', 'i', 'c', 's', 'e']
else:
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

print(letters)
N_letters = len(letters)
# Numbers
numbers = np.arange(N_letters) + 1
# Linearly ordered real-world concepts
realworld_linear = [['cold', 'cool', 'warm', 'hot'],
                    ['love', 'like', 'dislike', 'hate'],
                    ['jack', 'queen', 'king', 'ace'],
                    ['penny', 'nickel', 'dime', 'quarter'],
                    ['second', 'minute', 'hour', 'day']]
def get_letter_with_interval(letter, dir=1):
    ind = letters.index(letter)
    ind_new = (ind+dir*INTERVAL_MOD) % N_letters
    return letters[ind_new]
# Successor transformation

def apply_succ(prob_letters, modifications):
    last_letter = prob_letters[-1]
    if 'larger_int_focus' in modifications:
        last_letter_new = get_letter_with_interval(last_letter)
        #print(last_letter_new, 'instead of', last_letter)
        last_letter = last_letter_new
    return [prob_letters[:-1], prob_letters[:-2] + [last_letter]]

# Predecessor transformation
def apply_pred(prob_letters, modifications):
    first_letter = prob_letters[0]
    if 'larger_int_focus' in modifications:
        first_letter_new = get_letter_with_interval(first_letter, dir=-1)
        first_letter = first_letter_new
    return [prob_letters[1:], [first_letter] + prob_letters[2:]]

# Add letter to sequence
def apply_add_letter(prob_letters, modifications):
    last_letter = prob_letters[-1]
    if 'larger_int_focus' in modifications:
        last_letter_new = get_letter_with_interval(last_letter)
        last_letter = last_letter_new
    return [prob_letters[:-1], prob_letters[:-1] + [last_letter]]

# Remove redundant letter
def apply_remove_redundant(prob_letters):
    redundant_loc = np.arange(len(prob_letters))
    np.random.shuffle(redundant_loc)
    redundant_loc = redundant_loc[0]
    prob_redundant = deepcopy(prob_letters)
    prob_redundant.insert(redundant_loc, prob_letters[redundant_loc])
    return [prob_redundant, prob_letters]

# Remove out-of-place character
def apply_fix_alphabet(prob_letters):
    remaining_letters = np.array(deepcopy(letters))
    remaining_letters = remaining_letters[np.all(np.expand_dims(np.array(remaining_letters),1) != np.expand_dims(np.array(prob_letters),0), 1)]
    np.random.shuffle(remaining_letters)
    insert_letter = remaining_letters[0]
    insert_loc = np.arange(len(prob_letters))
    np.random.shuffle(insert_loc)
    insert_loc = insert_loc[0]
    prob_letters_insert = deepcopy(prob_letters)
    prob_letters_insert[insert_loc] = insert_letter
    return [prob_letters_insert, prob_letters]

# Sort letters
def apply_sort(prob_letters):
    #prob_letters.sort()
    swap_loc = np.arange(len(prob_letters))
    np.random.shuffle(swap_loc)
    i_loc = swap_loc[0]
    j_loc = swap_loc[1]
    i_letter = prob_letters[i_loc]
    j_letter = prob_letters[j_loc]
    prob_swapped = deepcopy(prob_letters)
    prob_swapped[i_loc] = j_letter
    prob_swapped[j_loc] = i_letter
    return [prob_swapped, prob_letters]

def apply_succ_pred(prob_letters, modifications):
    # Apply successor, then predecessor on the resulting pair
    succ_out = apply_succ(prob_letters[1:], modifications)
    for seq in succ_out:
        seq.insert(0, prob_letters[0])
    # succ_out = [A, B]; apply pred independently to each
    A2 = apply_pred(succ_out[0], modifications)[0]
    B2 = apply_pred(succ_out[1], modifications)[1]
    return [A2, B2]


def apply_fix_alphabet_add_letter(prob_letters, modifications):
    # Apply fix_alphabet, then add_letter
    prob_letters = prob_letters[1:] # So sequence length is right
    fix_out = apply_fix_alphabet(prob_letters)
    # fix_out = [A, B]; apply add_letter independently
    A2 = apply_add_letter(fix_out[0], modifications)[0]
    B2 = apply_add_letter(fix_out[1], modifications)[1]
    return [A2, B2]


def apply_sort_succ(prob_letters, modifications):
    # Apply sort, then successor
    prob_letters = prob_letters[1:]
    sort_out = apply_sort(prob_letters)

    last_letter = prob_letters[-1]
    if 'larger_int_focus' in modifications:
        last_letter_new = get_letter_with_interval(last_letter)
        #print(last_letter_new, 'instead of', last_letter)
        last_letter = last_letter_new
    sort_out[0].remove(last_letter)
    return [sort_out[0], sort_out[1][:-2] + [last_letter]]

# Method for generating subset of problems. standard_len modified to 6 for composite problems
def gen_prob_subset(N_generalize=0, N_prob=100, standard_len=6, longer_targ_len=9, larger_int_size=INTERVAL_SIZE,
                    trans_allowed=['succ', 'pred', 'add_letter', 'remove_redundant', 'fix_alphabet', 'sort', 
                                    'succ_pred', 'fix_alphabet_add_letter', 'sort_succ'],
                    gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse', 'realworld'],
                    modifications=[]):
    # Initialize storage for problems
    all_prob = []
    all_trans = []
    all_gen = []
    all_src_letters = []
    all_tgt_letters = []
    all_mods = []
    while len(all_prob) < N_prob:
        # Sample source letters
        if 'larger_int_all' in modifications:
            src_span = (standard_len * larger_int_size) - 1
            src_start = np.floor(np.random.rand() * (len(letters)-src_span)).astype(int)
            src_letters = letters[src_start:src_start + src_span][::larger_int_size]
        else:
            src_span = standard_len + INTERVAL_MOD -1
            src_start = np.floor(np.random.rand() * (len(letters)-src_span)).astype(int)
            src_letters = letters[src_start:src_start+standard_len]
        # Sample generalizations
        random.shuffle(gen_allowed)
        generalize = gen_allowed[:N_generalize]
        # Sample target letters
        if 'realworld' in generalize:
            random.shuffle(realworld_linear)
            tgt_letters = realworld_linear[0]
        else:
            if 'longer_targ' in generalize and 'larger_int' in generalize:
                tgt_span = (longer_targ_len * larger_int_size) - 1
                src_duplicate = True
                while src_duplicate:
                    tgt_start = np.floor(np.random.rand() * (len(letters)-(tgt_span-1))).astype(int)
                    if src_start != tgt_start:
                        src_duplicate = False
                tgt_letters = letters[tgt_start:tgt_start+tgt_span][::2]
            elif 'longer_targ' in generalize and 'larger_int' not in generalize:
                src_duplicate = True
                while src_duplicate:
                    tgt_start = np.floor(np.random.rand() * (len(letters)-(longer_targ_len-1))).astype(int)
                    if src_start != tgt_start:
                        src_duplicate = False
                tgt_letters = letters[tgt_start:tgt_start+longer_targ_len]
            elif 'larger_int_all' in modifications or ('longer_targ' not in generalize and 'larger_int' in generalize):
                tgt_span = standard_len * larger_int_size - 1
                if 'larger_int_all' in modifications:
                    assert src_span == tgt_span
                src_duplicate = True
                while src_duplicate:
                    tgt_start = np.floor(np.random.rand() * (len(letters)-(tgt_span))).astype(int) # skewing towards the beginning of the alphabet
                    if src_start != tgt_start:
                        src_duplicate = False
                tgt_letters = letters[tgt_start:tgt_start+tgt_span][::larger_int_size]
            elif 'longer_targ' not in generalize and 'larger_int' not in generalize:
                src_duplicate = True
                while src_duplicate:
                    tgt_start = np.floor(np.random.rand() * (len(letters)-(standard_len-1))).astype(int)
                    if src_start != tgt_start:
                        src_duplicate = False
                tgt_letters = letters[tgt_start:tgt_start+standard_len]
        # Reverse target letters
        if 'reverse' in generalize:
            tgt_letters.reverse()
        # Sample transformation
        random.shuffle(trans_allowed)
        trans = trans_allowed[0]
        # Apply transformation
        if trans == 'succ':
            src = apply_succ(src_letters, modifications)  # apply_succ(src_letters, do_modify)
            tgt = apply_succ(tgt_letters, modifications) # apply_succ(tgt_letters, do_modify)
        elif trans == 'pred':
            src = apply_pred(src_letters, modifications)
            tgt = apply_pred(tgt_letters, modifications)
        elif trans == 'add_letter':
            src = apply_add_letter(src_letters, modifications)
            tgt = apply_add_letter(tgt_letters, modifications)
        elif trans == 'remove_redundant':
            src = apply_remove_redundant(src_letters)
            tgt = apply_remove_redundant(tgt_letters)
        elif trans == 'fix_alphabet':
            src = apply_fix_alphabet(src_letters)
            tgt = apply_fix_alphabet(tgt_letters)
        elif trans == 'sort':
            src = apply_sort(src_letters)
            tgt = apply_sort(tgt_letters)
        elif trans == 'succ_pred':
            src = apply_succ_pred(src_letters, modifications)
            tgt = apply_succ_pred(tgt_letters, modifications)
        elif trans == 'fix_alphabet_add_letter':
            src = apply_fix_alphabet_add_letter(src_letters, modifications)
            tgt = apply_fix_alphabet_add_letter(tgt_letters, modifications)
        elif trans == 'sort_succ':
            src = apply_sort_succ(src_letters, modifications)
            tgt = apply_sort_succ(tgt_letters, modifications)
        # Generalization from letters to numbers
        if 'letter2num' in generalize:
            new_tgt = []
            for i in range(len(tgt)):
                new_tgt_i = []
                for j in range(len(tgt[i])):
                    new_tgt_i.append(numbers[np.where(np.array(letters) == tgt[i][j])[0][0]])
                new_tgt.append(new_tgt_i)
            tgt = new_tgt
        # Interleaved X's (or 0's, if target composed of numbers)
        if 'interleaved' in generalize:
            new_tgt = []
            for i in range(len(tgt)):
                new_tgt_i = []
                for j in range(len(tgt[i])):
                    new_tgt_i.append(tgt[i][j])
                    if 'letter2num' in generalize:
                        new_tgt_i.append('0')
                    else:
                        new_tgt_i.append('x')
                new_tgt.append(new_tgt_i)
            tgt = new_tgt
        # Grouping
        if 'group' in generalize:
            new_tgt = []
            for i in range(len(tgt)):
                new_tgt_i = []
                for j in range(len(tgt[i])):
                    new_tgt_i.append(tgt[i][j])
                    new_tgt_i.append(tgt[i][j])
                new_tgt.append(new_tgt_i)
            tgt = new_tgt
        # Check that problem doesn't already exist
        prob = [src, tgt]
        duplicate = False
        #if trans not in ['add_letter', 'remove_redundant', 'fix_alphabet', 'sort', 'pred']: # todo in pandas
        # using pandas instead of numpy due to dimension error
        for p_prev in range(len(all_prob)):
            if pd.Series(prob).shape == pd.Series(all_prob[p_prev]).shape:
                if np.all(pd.Series(prob) == pd.Series(all_prob[p_prev])):
                    print('duplicate', prob, all_prob[p_prev])
                    duplicate = True
        # Add to problem subset
        if not duplicate:
            all_prob.append(prob)
            all_trans.append(trans)
            all_gen.append(generalize)
            all_src_letters.append(src_letters)
            all_tgt_letters.append(tgt_letters)
            all_mods.append(modifications)

    return {'prob': all_prob, 'trans': all_trans, 'gen': all_gen, 'src_letters': all_src_letters, 'tgt_letters': all_tgt_letters,
            'modifications': all_mods}

# Add problems to json and numpy file
def save_prob(all_prob, prob_type_name, all_prob_js):
    # Convert to strings and save as json
    all_data = []
    for p in range(len(all_prob['prob'])):
        # A
        prompt = '['
        for i in range(len(all_prob['prob'][p][0][0])):
            prompt += str(all_prob['prob'][p][0][0][i])
            if i < len(all_prob['prob'][p][0][0]) - 1:
                prompt += ' '
        prompt += '] &nbsp ['
        # B
        for i in range(len(all_prob['prob'][p][0][1])):
            prompt += str(all_prob['prob'][p][0][1][i])
            if i < len(all_prob['prob'][p][0][1]) - 1:
                prompt += ' '
        prompt += ']<br>['
        # C
        for i in range(len(all_prob['prob'][p][1][0])):
            prompt += str(all_prob['prob'][p][1][0][i])
            if i < len(all_prob['prob'][p][1][0]) - 1:
                prompt += ' '
        prompt += '] &nbsp [&nbsp ? &nbsp]'
        # Add to dataset
        all_data.append({'prompt': prompt, 'prob_ind': p})
    # Add to javascript data
    all_prob_js[prob_type_name] = all_data
    return all_prob_js

# Split subset
def split_subset(all_prob, N_split):
    all_prob_split = []
    N_subset = int(len(all_prob['prob']) / N_split)
    for s in range(N_split):
        subset = {}
        for key in all_prob.keys():
            subset[key] = []
        for p in range(N_subset*s,N_subset*(s+1)):
            for key in all_prob.keys():
                subset[key].append(all_prob[key][p])
        all_prob_split.append(subset)
    return all_prob_split

def main(args):
    if args.modified or args.synthetic:
        do_modify = True
    else:
        do_modify = False
    if do_modify:
        suffix = get_suffix_modified(args)
    else:
        suffix = ''

    f_name = f'all_prob_composite{suffix}.npz'
    fp = './' + f_name
    if suffix == '_modified_prompt':  # copy modified file since same problems
        fp_modified = './' + f'all_prob_modified.npz'
        shutil.copyfile(fp_modified, fp)
        vrs_dir = get_version_dir(args)
        shutil.copyfile(fp_modified, os.path.join(vrs_dir, f_name))
        print('Use same problems as without prompt. Copied to', fp)
        return

    if do_modify:
        modifications_1 = ['larger_int_focus']
        modifications_2 = ['larger_int_all']
        # Generate all basic analogies (zero generalizations)
        # all_succ = gen_prob_subset(trans_allowed=['succ'], modifications=modifications_1)
        # all_pred = gen_prob_subset(trans_allowed=['pred'], modifications=modifications_1)
        # all_add_letter = gen_prob_subset(trans_allowed=['add_letter'], modifications=modifications_1)
        # all_remove_redundant = gen_prob_subset(trans_allowed=['remove_redundant'], modifications=modifications_2)
        # all_fix_alphabet = gen_prob_subset(trans_allowed=['fix_alphabet'], modifications=modifications_2)
        # all_sort = gen_prob_subset(trans_allowed=['sort'], modifications=modifications_2)
        all_succ_pred = gen_prob_subset(trans_allowed=['succ_pred'], modifications=modifications_1)
        all_fix_alphabet_add_letter = gen_prob_subset(trans_allowed=['fix_alphabet_add_letter'], modifications=modifications_1)
        all_sort_succ = gen_prob_subset(trans_allowed=['sort_succ'], modifications=modifications_1)

        all_prob_types = [all_succ_pred, all_fix_alphabet_add_letter, all_sort_succ]
        all_prob_type_names = ['succ_pred', 'fix_alphabet_add_letter', 'sort_succ']
    else:
        # Generate all basic analogies (zero generalizations)
        # all_succ = gen_prob_subset(trans_allowed=['succ'])
        # all_pred = gen_prob_subset(trans_allowed=['pred'])
        # all_add_letter = gen_prob_subset(trans_allowed=['add_letter'])
        # all_remove_redundant = gen_prob_subset(trans_allowed=['remove_redundant'])
        # all_fix_alphabet = gen_prob_subset(trans_allowed=['fix_alphabet'], gen_allowed=['larger_int'])
        # all_sort = gen_prob_subset(trans_allowed=['sort'], gen_allowed=['larger_int'])
        all_succ_pred = gen_prob_subset(trans_allowed=['succ_pred'])
        all_fix_alphabet_add_letter = gen_prob_subset(trans_allowed=['fix_alphabet_add_letter'])
        all_sort_succ = gen_prob_subset(trans_allowed=['sort_succ'])

        # Generate all problems with one generalization (randomly sample transformations)
        all_larger_int = gen_prob_subset(N_generalize=1, gen_allowed=['larger_int'])
        all_longer_targ = gen_prob_subset(N_generalize=1, gen_allowed=['longer_targ'])
        all_group = gen_prob_subset(N_generalize=1, gen_allowed=['group'])
        all_interleaved = gen_prob_subset(N_generalize=1, gen_allowed=['interleaved'])
        all_letter2num = gen_prob_subset(N_generalize=1, gen_allowed=['letter2num'])
        all_reverse = gen_prob_subset(N_generalize=1, gen_allowed=['reverse'])

        # Generate all problems with 2 and 3 generalizations
        all_2gen = gen_prob_subset(N_generalize=2, N_prob=600, gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse'])
        all_2gen_split = split_subset(all_2gen, 6)
        all_3gen = gen_prob_subset(N_generalize=3, N_prob=600, gen_allowed=['larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse'])
        all_3gen_split = split_subset(all_3gen, 6)

        # Generate problems involving generalization to real-world concepts
        all_realworld_succ = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['succ'], gen_allowed=['realworld'])
        all_realworld_pred = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['pred'], gen_allowed=['realworld'])
        all_realworld_add_letter = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['add_letter'], gen_allowed=['realworld'])
        all_realworld_sort = gen_prob_subset(standard_len=4, N_generalize=1, trans_allowed=['sort'], gen_allowed=['realworld'])

        # Combine problems
        all_prob_types = [all_succ_pred, all_fix_alphabet_add_letter, all_sort_succ,
                          all_larger_int, all_longer_targ, all_group, all_interleaved, all_letter2num, all_reverse,
                          all_2gen_split[0], all_2gen_split[1], all_2gen_split[2], all_2gen_split[3], all_2gen_split[4], all_2gen_split[5],
                          all_3gen_split[0], all_3gen_split[1], all_3gen_split[2], all_3gen_split[3], all_3gen_split[4], all_3gen_split[5],
                          all_realworld_succ, all_realworld_pred, all_realworld_add_letter, all_realworld_sort]
        all_prob_type_names = ['succ_pred', 'fix_alphabet_add_letter', 'sort_succ',
                               'larger_int', 'longer_targ', 'group', 'interleaved', 'letter2num', 'reverse',
                               '2gen_split1', '2gen_split2', '2gen_split3', '2gen_split4', '2gen_split5', '2gen_split6',
                               '3gen_split1', '3gen_split2', '3gen_split3', '3gen_split4', '3gen_split5', '3gen_split6',
                               'realworld_succ', 'realworld_pred', 'realworld_add_letter', 'realworld_sort']


    # Create js variable for all_problems
    all_prob_js = {}
    all_prob_np = {}
    for p in range(len(all_prob_types)):
        all_prob_js = save_prob(all_prob_types[p], all_prob_type_names[p], all_prob_js)
        all_prob_np[all_prob_type_names[p]] = all_prob_types[p]
    # Convert to json strings and write to json
    # all_prob_json_string = json.dumps(all_prob_js)
    # js_fname = f'./all_prob{suffix}.js'
    # js_fid = open(js_fname, 'w')
    # js_fid.write('var all_problems = ' + all_prob_json_string)
    # js_fid.close()
    save_to_versions(args, all_prob_np, f_name)


def save_to_versions(args, all_prob_np, f_name):
    vrs_dir = get_version_dir(args)
    fp = os.path.join(vrs_dir, f_name)
    np.savez(fp, all_prob=all_prob_np)


if __name__ == '__main__':
    main(args)
