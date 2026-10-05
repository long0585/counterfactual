#!/bin/bash
# Standalone retry pass for glm and minimax ONLY.
#
# Why this exists: in the original hf_models_pipeline.sh 4-way parallel run,
# kimi and deepseek are seeing only occasional 504s and recovering fine, but
# glm and minimax are hitting persistent 504 Gateway Time-outs from the HF
# router (glm.log alone logged 255 straight 504s and exhausted the original
# script's 10-attempt/10s-fixed-delay retry cap). Since kimi/deepseek keep
# succeeding under the same shared gateway, this looks like degradation
# specific to the GLM-5.3 / MiniMax-M3 backend deployments on DeepInfra
# rather than pure concurrency overload -- so this script does NOT touch
# kimi/deepseek (leave the original hf_models_pipeline.sh run for those
# alone) and only re-targets the two struggling providers, with a larger
# retry budget to tolerate a longer blip.
#
# Safe to run concurrently with the original hf_models_pipeline.sh job:
# it only launches glm/minimax (whose jobs in the original run have
# already given up and exited), writes to distinctly-named log files
# (logs/glm_retry.log, logs/minimax_retry.log) to avoid clobbering the
# original logs/glm.log / logs/minimax.log, and relies on test_gpt.py's
# per-trial checkpointing to safely resume each provider's existing
# checkpoint rather than redoing completed trials.
#
# Retry budget: 20 attempts per command (vs. the original script's 10),
# with exponential backoff starting at 15s and capped at 90s between
# attempts (vs. the original's fixed 10s) -- meant to comfortably outlast
# a transient backend outage without hammering the gateway or looping
# forever on a truly persistent failure.

set -uo pipefail
cd "$(dirname "$0")"

FORWARD="a b c d e f g h i j k l m n o p q r s t u v w x y z"
BACKWARD="z y x w v u t s r q p o n m l k j i h g f e d c b a"
RANDOM_ALPHABET="x y l k w b f z t n j r q a h v g m u o p d i c s e"
ALPHABETS=("$FORWARD" "$BACKWARD" "$RANDOM_ALPHABET")

PROVIDERS=(glm minimax)

# Effort per provider (mirrors hf_models_pipeline.sh). Empty = omit --effort
# entirely (minimax has reasoning_style="none" and rejects it).
effort_for_provider() {
    case "$1" in
        glm) echo "low" ;;
        minimax) echo "" ;;
    esac
}

# Resolved default model ID per provider (must match engine/config.py's
# PROVIDER_PROFILES default_model -- see hf_models_pipeline.sh for the
# same requirement/rationale).
model_for_provider() {
    case "$1" in
        glm) echo "zai-org/GLM-5.3:deepinfra" ;;
        minimax) echo "MiniMaxAI/MiniMax-M3:deepinfra" ;;
    esac
}

TRIALS=100

MAX_RETRY_ATTEMPTS=20
BACKOFF_START=15
BACKOFF_CAP=90
run_with_retry() {
    local attempt=1
    local delay=$BACKOFF_START
    until "$@"; do
        if (( attempt >= MAX_RETRY_ATTEMPTS )); then
            echo "[$(date +%H:%M:%S)] Command failed $MAX_RETRY_ATTEMPTS times in a row -- giving up (this looks like a persistent failure, not a transient one): $*"
            return 1
        fi
        echo "[$(date +%H:%M:%S)] Command failed (attempt $attempt/$MAX_RETRY_ATTEMPTS) -- retrying in ${delay}s (progress is checkpointed, so this resumes rather than restarting): $*"
        sleep "$delay"
        attempt=$((attempt + 1))
        delay=$(( delay * 2 ))
        if (( delay > BACKOFF_CAP )); then
            delay=$BACKOFF_CAP
        fi
    done
}

# Bash-3.2-safe way to splice a possibly-empty array into a command line
# without set -u treating it as unset (the "${arr[@]:-}" idiom silently
# injects one stray empty-string argument when the array is empty, which
# argparse then rejects -- see hf_models_pipeline.sh's fix for the same bug).
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

    echo "[$(date +%H:%M:%S)] [$provider] done."
}

mkdir -p logs

pids=()
names=()
for provider in "${PROVIDERS[@]}"; do
    logfile="logs/${provider}_retry.log"
    echo "Starting $provider (log: $logfile)..."
    run_provider "$provider" > "$logfile" 2>&1 &
    pids+=("$!")
    names+=("$provider")
done

echo "Launched ${#pids[@]} provider(s) in parallel: ${names[*]}"
echo "Tail any of them with: tail -f logs/<provider>_retry.log"

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[${names[$i]}] SUCCEEDED"
    else
        echo "[${names[$i]}] FAILED -- see logs/${names[$i]}_retry.log"
        status=1
    fi
done

exit "$status"
