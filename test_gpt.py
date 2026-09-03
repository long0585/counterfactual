import numpy as np
import argparse
import json
import os
import time

from engine import ModelEngine, PROVIDER_PROFILES, ProviderConfig, result_model_label

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--interval_size', type=int, default=1, help='Interval size')
parser.add_argument('--gpt_engine', '--model', dest='gpt_engine', type=str, default=None,
                    help='Model ID (defaults to the selected provider profile)')
parser.add_argument('--provider', choices=sorted(PROVIDER_PROFILES), default='openai',
                    help='API provider profile')
parser.add_argument('--base_url', '--base-url', dest='base_url',
                    help='Override the provider endpoint')
parser.add_argument('--api_key_env', '--api-key-env', dest='api_key_env',
                    help='Name of the environment variable containing the API key')
parser.add_argument('--alphabet', type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z', help='Choose custom alphabet')
parser.add_argument('--effort', type=str, help='Provider-supported reasoning effort')
parser.add_argument('--max_output_tokens', '--max-output-tokens', dest='max_output_tokens',
                    type=int, help='Maximum generated tokens (mapped per provider)')
parser.add_argument('--extra_body_json', '--extra-body-json', dest='extra_body_json',
                    help='Provider-specific JSON object merged into extra_body')
parser.add_argument('--timeout', type=float, default=120.0, help='API timeout in seconds')
parser.add_argument('--max_retries', '--max-retries', dest='max_retries', type=int, default=3,
                    help='Maximum attempts per puzzle (default: 3)')
parser.add_argument('--retry_delay', '--retry-delay', dest='retry_delay', type=float, default=5.0,
                    help='Seconds between attempts')
parser.add_argument('--trials', type=int, default=10, help='Number of trials')
parser.add_argument('--composite', action='store_true', default=False)
args = parser.parse_args()

if args.max_retries < 1:
    parser.error('--max-retries must be at least 1')

try:
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
except json.JSONDecodeError as exc:
    parser.error(f'--extra-body-json must be valid JSON: {exc}')
if extra_body is not None and not isinstance(extra_body, dict):
    parser.error('--extra-body-json must decode to a JSON object')

try:
    provider_config = ProviderConfig.from_env(
        args.provider,
        model=args.gpt_engine,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
    )
except ValueError as exc:
    parser.error(str(exc))

args.gpt_engine = provider_config.model
client = ModelEngine(provider_config)

alphabet_suffix = args.alphabet.replace(" ", "")
effort_level = args.effort

# Load all problems
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
prob_types = list(all_prob.item().keys())
if args.composite:
    prob_types = prob_types[:3]
else:
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
        completion = None
        for attempt in range(1, args.max_retries + 1):
            try:
                result = client.generate(
                    prompt,
                    effort=effort_level,
                    max_output_tokens=args.max_output_tokens,
                    extra_body=extra_body,
                )
                response = result.text
                completion = result.raw_response
                if response:
                    break
                raise ValueError('Provider returned an empty final answer')
            except Exception as e:
                if attempt == args.max_retries:
                    raise RuntimeError(
                        f'API request failed after {args.max_retries} attempts'
                    ) from e
                print(f'Error: {e}. Retrying ({attempt}/{args.max_retries})...')
                time.sleep(args.retry_delay)
        
        print(response)
        prob_type_responses.append(response)
        prob_type_completions.append(completion)
    all_prob_type_responses.append(prob_type_responses)
    all_prob_type_completions.append(prob_type_completions)

avg_time = (time.time() - start_time) / (N_trials_per_prob_type * N_prob_types) # avg time/problem in seconds
# Save results. Historical OpenAI names are preserved; provider model IDs containing
# slashes/colons are made safe as a single filename component.
output_dir = './test_outputs_composite' if args.composite else './test_outputs'
os.makedirs(output_dir, exist_ok=True)
model_label = result_model_label(args.provider, args.gpt_engine, effort_level)
save_fname = os.path.join(
    output_dir,
    model_label + '_' + alphabet_suffix + '_int' + str(args.interval_size) + '_results.npz',
)
np.savez(save_fname, all_prob_type_responses=all_prob_type_responses, all_prob_type_completions=all_prob_type_completions, avg_time=avg_time)
