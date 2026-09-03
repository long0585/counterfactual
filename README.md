# Counterfactual alphabet evaluation

This repository generates synthetic alphabet-transformation puzzles, sends them to a model API, saves raw and normalized results, and analyzes accuracy.

## Architecture and call flow

```text
gen_problems.py -> all_prob_int{1,2}/*.npz
                         |
                         v
test_gpt.py -> prompt construction -> engine/ModelEngine
                                      |-- OpenAI Responses API
                                      |-- Hugging Face Responses API (beta)
                                      |   `-- Qwen / GLM / MiniMax / Kimi / DeepSeek
                                      `-- custom OpenAI-compatible Chat Completions
                         |
                         v
                 test_outputs/*.npz -> analyze_gpt.py -> plots/metrics
```

The experiment loops, prompt text, problem selection, result arrays, and analysis pipeline remain unchanged. `engine/` owns only provider configuration, request differences, and response normalization.

## Setup

Use Python 3.9 or newer, create a virtual environment, then install:

```bash
pip install -r requirements.txt
```

API keys are read only from environment variables. `.env.example` lists supported names; the code does not load `.env` files or contain secrets.

## Providers and current default models (verified 2026-09-02)

| Provider | Default model | Key variable | Endpoint |
|---|---|---|---|
| `openai` | `gpt-4o` (backward-compatible default) | `OPENAI_API_KEY` | OpenAI default |
| `qwen` | `Qwen/Qwen3.8-27B:deepinfra` | `HF_TOKEN` | HF Router |
| `glm` | `zai-org/GLM-5.3:deepinfra` | `HF_TOKEN` | HF Router |
| `minimax` | `MiniMaxAI/MiniMax-M3:deepinfra` | `HF_TOKEN` | HF Router |
| `kimi` | `moonshotai/Kimi-K3:deepinfra` | `HF_TOKEN` | HF Router |
| `deepseek` | `deepseek-ai/DeepSeek-V4-Pro:deepinfra` | `HF_TOKEN` | HF Router |
| `huggingface` | `Qwen/Qwen3.8-27B:deepinfra` | `HF_TOKEN` | HF Router |
| `custom` | required | `MODEL_API_KEY` | `MODEL_BASE_URL` |

The five named model profiles are logical aliases over one Hugging Face connection.
They no longer read vendor keys or call vendor endpoints. `deepinfra` was live for
all five model IDs when verified, so it is pinned to make research runs more
comparable than the dynamic `:fastest` policy. Re-check availability before a
large run because hosted providers can change.

Examples (PowerShell key setup shown; do not put tokens in source files):

```powershell
$env:HF_TOKEN = "hf_your-token"

python test_gpt.py --provider qwen --effort xhigh --trials 10
python test_gpt.py --provider glm --trials 10
python test_gpt.py --provider minimax --trials 10
python test_gpt.py --provider kimi --effort max --trials 10
python test_gpt.py --provider deepseek --effort high --trials 10

# Generic HF profile for any currently hosted conversational model:
python test_gpt.py --provider huggingface --model "Qwen/Qwen3.8-27B:fastest" --trials 10
```

Hugging Face model IDs can be switched without code changes, for example:

- `Qwen/Qwen3.8-27B`
- `zai-org/GLM-5.3`
- `MiniMaxAI/MiniMax-M3`
- `moonshotai/Kimi-K3`
- `deepseek-ai/DeepSeek-V4-Pro`

Router availability is dynamic. Query Hugging Face's `/v1/models` endpoint or use its playground before a paid run; a Hub repository existing does not guarantee that every inference provider currently serves it.

The aliases select the open model repositories shown above. They are not aliases
for vendor-only products such as `qwen3.8-max`.

### Reasoning behavior through HF Responses

- Every named HF profile uses `reasoning={"effort": ...}` and `max_output_tokens` when applicable.
- Qwen accepts `low`, `medium`, or `xhigh` in this configuration.
- MiniMax has no locally validated effort mapping, so omit `--effort`.
- Kimi and DeepSeek accept the effort values listed in `engine/config.py`.
- GLM effort values are passed through to HF; omit `--effort` if the selected backing provider rejects a value.
- Vendor-specific fields such as `thinking.type`, `reasoning_split`, and `max_completion_tokens` are not sent through the HF Responses endpoint.
- Override a model with a suffix such as `:fastest`, `:cheapest`, or a concrete provider. Avoid dynamic policies in a final benchmark.

Override any default with `--model`, `--base-url`, or `--api-key-env`. The last option is the *name* of an environment variable, never the key value itself.

## Offline validation

No API key and no billable call are needed for:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

The tests mock the SDK boundary and verify that all five aliases use `HF_TOKEN`,
the HF Router, Responses request parameters, and normalized output text/reasoning.
