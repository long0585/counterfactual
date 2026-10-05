#!/bin/bash
# Pipeline: re-run the counterfactual letter-sequence tests on the
# Hugging-Face-gateway-routed models (Qwen, GLM, MiniMax, Kimi, DeepSeek),
# mirroring the pattern in gpt5-4_script.sh / gpt5-4_composite.sh.
#
# Scope: trials=100, forward/backward/random alphabets only, effort held at
# "low" per model (or the closest supported tier -- see engine/config.py
# and README.md for the per-provider effort menus). Regular problems run at
# both interval sizes 1 and 2; composite problems only run at interval 1
# (composite problem sets are intentionally not generated for interval 2).
#
# Requires HF_TOKEN to be set in the environment.
#
# Each test_gpt.py call (raw-response generation) is immediately followed by
# an analyze_gpt.py call (scoring), mirroring gpt5-4_composite.sh's pattern
# for GPT-5. analyze_gpt.py computes accuracy itself from the raw responses
# in test_outputs/ against the ground-truth problems in all_prob_int{1,2}/ --
# no separate scoring step or per-provider "_alphabets.txt" file is required
# for this. (A "<label>_alphabets.txt" file is only needed by the optional
# analyze_all_alphabets.py multi-alphabet sweep script; see the accompanying
# *_alphabets.txt files generated alongside this script for that purpose.)
# --gpt_engine is set explicitly to each provider's resolved default model
# ID so analyze_gpt.py's result_model_label() matches the filename test_gpt.py
# actually wrote (test_gpt.py resolves the same default when --gpt_engine is
# omitted, per engine/config.py's PROVIDER_PROFILES).
#
# Parallelism: each provider's full sweep (2 intervals x 3 alphabets = 6
# test_gpt.py/analyze_gpt.py pairs, ~600 trials each) takes several hours
# end to end, dominated by network/generation wait time rather than CPU. So
# instead of running providers one after another, run_provider() is launched
# as a background job per provider and they proceed concurrently -- total
# wall time is roughly the time for ONE provider's sweep instead of N of
# them. Each provider writes its own log file under ./logs/ since stdout
# would otherwise interleave across the background jobs. test_gpt.py's
# per-trial checkpointing (see test_gpt.py) makes this safe: if the shared
# HF gateway gets extra load from running everything at once and a call
# 504s, the retry loop below just resumes that provider's run rather than
# losing progress. If concurrent 504/429s turn out to be a real problem in
# practice, reduce PROVIDERS to fewer entries and run the rest in a second
# pass.
#
# Qwen has already been run to completion separately (see quick.sh) and is
# intentionally excluded from PROVIDERS below to avoid re-running (and
# overwriting) 600 already-completed trials per alphabet/interval. Re-add it
# to PROVIDERS if you ever need to regenerate it from scratch.

set -uo pipefail
cd "$(dirname "$0")"

FORWARD="a b c d e f g h i j k l m n o p q r s t u v w x y z"
BACKWARD="z y x w v u t s r q p o n m l k j i h g f e d c b a"
RANDOM_ALPHABET="x y l k w b f z t n j r q a h v g m u o p d i c s e"
ALPHABETS=("$FORWARD" "$BACKWARD" "$RANDOM_ALPHABET")

# Remaining providers to run (Qwen already completed -- see comment above).
PROVIDERS=(glm minimax kimi deepseek)

# Effort per provider: "low" where supported, otherwise the closest
# available tier per engine/config.py's supported_efforts. Empty = omit
# --effort entirely (MiniMax has reasoning_style="none" and rejects it).
# All four HF profiles besides MiniMax accept "low" directly.
# (Using a case statement instead of an associative array since macOS ships
# bash 3.2 by default, which doesn't support `declare -A`.)
effort_for_provider() {
    case "$1" in
        qwen) echo "low" ;;
        glm) echo "low" ;;
        minimax) echo "" ;;
        kimi) echo "low" ;;
        deepseek) echo "low" ;;
    esac
}

# Resolved default model ID per provider (must match engine/config.py's
# PROVIDER_PROFILES default_model, since that's what test_gpt.py resolves to
# when --gpt_engine is omitted). analyze_gpt.py has no provider-resolution
# logic of its own, so we pass this explicitly to keep its result_model_label()
# output aligned with the filename test_gpt.py actually wrote.
model_for_provider() {
    case "$1" in
        qwen) echo "Qwen/Qwen3.8-27B:deepinfra" ;;
        glm) echo "zai-org/GLM-5.3:deepinfra" ;;
        minimax) echo "MiniMaxAI/MiniMax-M3:deepinfra" ;;
        kimi) echo "moonshotai/Kimi-K3:deepinfra" ;;
        deepseek) echo "deepseek-ai/DeepSeek-V4-Pro:deepinfra" ;;
    esac
}

TRIALS=100

# test_gpt.py checkpoints progress after every trial, so on a transient
# provider error (e.g. a 504 Gateway Timeout) it's safe to just retry the
# same command -- it resumes from the last completed trial instead of
# starting over. Without this, a crash would leave test_outputs stale and
# the immediately-following analyze_gpt.py call would silently score
# old/incomplete data (this bit us on the Qwen backward/interval-2 run).
#
# Capped at MAX_RETRY_ATTEMPTS: not every failure is transient. A model that
# consistently returns an empty final answer (e.g. truncating before it gets
# to the answer) will fail the exact same way on every retry -- looping
# forever would just burn hours and API calls without ever making progress.
# After the cap, give up and let set -e in run_provider mark this provider
# FAILED so the problem surfaces instead of hanging silently.
MAX_RETRY_ATTEMPTS=10
run_with_retry() {
    local attempt=1
    until "$@"; do
        if (( attempt >= MAX_RETRY_ATTEMPTS )); then
            echo "[$(date +%H:%M:%S)] Command failed $MAX_RETRY_ATTEMPTS times in a row -- giving up (this looks like a persistent failure, not a transient one): $*"
            return 1
        fi
        echo "[$(date +%H:%M:%S)] Command failed (attempt $attempt/$MAX_RETRY_ATTEMPTS) -- retrying in 10s (progress is checkpointed, so this resumes rather than restarting): $*"
        sleep 10
        attempt=$((attempt + 1))
    done
}

# --- Per-provider sweep --------------------------------------------------
# Regular: 1 provider x 2 intervals x 3 alphabets = 6 test+analyze pairs.
run_provider() {
    local provider="$1"
    set -e

    local effort_flag=()
    local effort
    effort="$(effort_for_provider "$provider")"
    if [[ -n "$effort" ]]; then
        effort_flag=(--effort "$effort")
    fi
    local model
    model="$(model_for_provider "$provider")"

    for interval in 1 2; do
        for alphabet in "${ALPHABETS[@]}"; do
            echo "[$(date +%H:%M:%S)] [$provider] interval=$interval alphabet='$alphabet' -- generating..."
            run_with_retry python3 ./test_gpt.py --provider "$provider" --alphabet "$alphabet" \
                --interval_size "$interval" --trials "$TRIALS" "${effort_flag[@]+"${effort_flag[@]}"}"
            echo "[$(date +%H:%M:%S)] [$provider] interval=$interval alphabet='$alphabet' -- scoring..."
            python3 ./analyze_gpt.py --provider "$provider" --gpt_engine "$model" \
                --alphabet "$alphabet" --interval_size "$interval" "${effort_flag[@]+"${effort_flag[@]}"}"
        done
    done
    # Disabling composite for now due to time constraint
    #
    # for interval in 1; do
    #     for alphabet in "${ALPHABETS[@]}"; do
    #         echo "[$(date +%H:%M:%S)] [$provider] composite interval=$interval alphabet='$alphabet' -- generating..."
    #         run_with_retry python3 ./test_gpt.py --provider "$provider" --alphabet "$alphabet" \
    #             --interval_size "$interval" --trials "$TRIALS" "${effort_flag[@]+"${effort_flag[@]}"}" --composite
    #         echo "[$(date +%H:%M:%S)] [$provider] composite interval=$interval alphabet='$alphabet' -- scoring..."
    #         python3 ./analyze_gpt.py --provider "$provider" --gpt_engine "$model" \
    #             --alphabet "$alphabet" --interval_size "$interval" "${effort_flag[@]+"${effort_flag[@]}"}" --composite
    #     done
    # done

    echo "[$(date +%H:%M:%S)] [$provider] done."
}

# --- Main: launch one background job per provider -------------------------
mkdir -p logs

pids=()
names=()
for provider in "${PROVIDERS[@]}"; do
    logfile="logs/${provider}.log"
    echo "Starting $provider (log: $logfile)..."
    run_provider "$provider" > "$logfile" 2>&1 &
    pids+=("$!")
    names+=("$provider")
done

echo "Launched ${#pids[@]} provider(s) in parallel: ${names[*]}"
echo "Tail any of them with: tail -f logs/<provider>.log"

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[${names[$i]}] SUCCEEDED"
    else
        echo "[${names[$i]}] FAILED -- see logs/${names[$i]}.log"
        status=1
    fi
done

exit "$status"
