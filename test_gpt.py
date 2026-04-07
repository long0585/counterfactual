from openai import OpenAI
import numpy as np
import builtins
import argparse
import time

# Set up OpenAI client
client = OpenAI()


# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--gpt_engine', type=str, default='gpt-4o', help='GPT-4 engine')
parser.add_argument('--alphabet', type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z', help='Choose custom alphabet')
parser.add_argument('--effort', type=str, help='Effort level for gpt5')
parser.add_argument('--trials', type=int, default=10, help='Number of trials')
args = parser.parse_args()

alphabet_suffix = args.alphabet.replace(" ", "")
effort_level = args.effort

# Load all problems
if args.interval_size == 1:
    all_prob = np.load('./all_prob_int1/all_prob_' + alphabet_suffix + '_interval1.npz', allow_pickle=True)['all_prob']
elif args.interval_size == 2:
    all_prob = np.load('./all_prob_int2/all_prob_' + alphabet_suffix + '_interval2.npz', allow_pickle=True)['all_prob']
prob_types = list(all_prob.item().keys())
prob_types = prob_types[:6]
N_prob_types = len(prob_types)

# Synthetic alphabet and prompt
custom_alphabet = args.alphabet
alphabet_prompt = "Let’s solve a puzzle problem involving the following fictional alphabet:\n\n[" + custom_alphabet + "]\n\nHere is the problem:\n\n"

# Evaluate
N_trials_per_prob_type = args.trials
all_prob_type_responses = []
all_prob_type_completions = []
start_time = time.time()
for p in range(N_prob_types):
    print('Problem type ' + str(p+1) + ' of ' + str(N_prob_types) + '...')
    prob_type_responses = []
    prob_type_completions = []
    for t in range(N_trials_per_prob_type):
        print('Trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...')
        # Generate prompt
        prob = all_prob.item()[prob_types[p]]['prob'][t]
        prompt = alphabet_prompt
        prompt += '[' + ' '.join(map(str, prob[0][0])) + '] [' + ' '.join(map(str, prob[0][1])) + ']\n'
        prompt += '[' + ' '.join(map(str, prob[1][0])) + '] [ ? ]\n\n'
        prompt += 'Please only provide the answer. Do not provide any additional explanation.\n\n'
        prompt += 'Answer:'

        print(prompt)
        # Get response
        response = ''
        while len(response) == 0:
            try:
                if not args.gpt_engine.startswith('gpt-5'):
                    completion = client.responses.create(
                                    model=args.gpt_engine,
                                    temperature=0,
                                    top_p=0,
                                    input=[{"role": "user", "content": prompt}]
                                    )
                else:
                    completion = client.responses.create(
                                    model=args.gpt_engine,
                                    reasoning={"effort": effort_level},
                                    input=[{"role": "user", "content": prompt}]
                                    )
                response = completion.output_text
            except Exception as e:
                print(f'Error: {e}. Trying again...')
                time.sleep(5)
        
        print(response)
        prob_type_responses.append(response)
        prob_type_completions.append(completion)
    all_prob_type_responses.append(prob_type_responses)
    all_prob_type_completions.append(prob_type_completions)

avg_time = (time.time() - start_time) / (N_trials_per_prob_type * N_prob_types) # avg time/problem in seconds
# Save results
if not args.gpt_engine.startswith("gpt-5"):
    save_fname = './test_outputs/' + args.gpt_engine + '_' + alphabet_suffix + '_int' + str(args.interval_size) + '_results.npz'
else:
    save_fname = './test_outputs/' + args.gpt_engine + '_' + effort_level + '_' + alphabet_suffix + '_int' + str(args.interval_size) + '_results.npz'
np.savez(save_fname, all_prob_type_responses=all_prob_type_responses, all_prob_type_completions=all_prob_type_completions, avg_time=avg_time)