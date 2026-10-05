#!/bin/bash
# Finish the remaining interval-1 gaps and all of interval-2 for
# deepseek/kimi/glm. Minimax is intentionally excluded (dropped per
# decision -- its backend has been too unreliable to be worth chasing).
#
# Why this script exists instead of re-running hf_models_pipeline.sh or
# retry_glm_minimax.sh: neither of those (nor test_gpt.py itself) checks
# whether a given (provider, interval, alphabet) combo already has a
# completed results.npz before starting -- they only check for a leftover
# checkpoint. So re-invoking them on an already-finished combo silently
# redoes all 100 trials from scratch. That's exactly what happened to glm:
# its backward-alphabet/interval-1 run had already completed successfully
# (checkpoint deleted, final results.npz written), but retry_glm_minimax.sh
# restarted it anyway and burned all 20 retry attempts hitting 504s on
# work that was already done -- it never even reached interval 2.
#
# The fix here is `already_complete()`, which computes the exact output
# filename test_gpt.py would use (via engine.result_model_label, the same
# function test_gpt.py itself calls -- not a re-implementation, so it can't
# drift out of sync) and skips straight past any combo whose results.npz
# already exists. Only genuinely incomplete or not-yet-started combos get
# a test_gpt.py/analyze_gpt.py call, and checkpointed-but-incomplete ones
# still resume normally instead of restarting.
#
# Current known state (informational; this script re-derives it live, so
# it stays correct even if you run it more than once):
#   deepseek: int1 all complete; int2 forward in-progress, backward/random not started
#   kimi:     int1 all complete; int2 forward complete, backward in-progress, random not started
#   glm:      int1 forward/backward complete, random in-progress; int2 nothing started
#
# Retry budget matches retry_glm_minimax.sh (20 attempts, exponential
# backoff 15s->90s cap) since glm/deepseek have both shown real gateway
# instability (deepseek recovers within budget; glm's failures turned out
# to be from wasted re-runs, not genuine unrecoverable errors, so the same
# budget should be more than sufficient now that wasted re-runs are gone).

set -uo pipefail
cd "$(dirname "$0")"

FORWARD="a b c d e f g h i j k l m n o p q r s t u v w x y z"
BACKWARD="z y x w v u t s r q p o n m l k j i h g f e d c b a"
RANDOM_ALPHABET="x y l k w b f z t n j r q a h v g m u o p d i c s e"
ALPHABETS=("$FORWARD" "$BACKWARD" "$RANDOM_ALPHABET")

# Minimax dropped -- see header comment.
PROVIDERS=(deepseek kimi glm)

effort_for_provider() {
    case "$1" in
        deepseek) echo "low" ;;
        kimi) echo "low" ;;
        glm) echo "low" ;;
    esac
}

model_for_provider() {
    case "$1" in
        deepseek) echo "deepseek-ai/DeepSeek-V4-Pro:deepinfra" ;;
        kimi) echo "moonshotai/Kimi-K3:deepinfra" ;;
        glm) echo "zai-org/GLM-5.3:deepinfra" ;;
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

# Returns success (0) if this (provider, interval, alphabet) combo already
# has a completed results.npz -- computed via the actual engine code
# test_gpt.py uses for its output filename, not a bash re-implementation,
# so this can't silently drift out of sync with test_gpt.py's real naming.
already_complete() {
    local provider="$1" interval="$2" alphabet="$3" model="$4" effort="$5"
    python3 - "$provider" "$model" "$effort" "$alphabet" "$interval" <<'EOF'
import os
import sys

from engine import result_model_label

provider, model, effort, alphabet, interval = sys.argv[1:6]
alphabet_suffix = alphabet.replace(" ", "")
label = result_model_label(provider, model, effort or None)
fname = os.path.join(
    "test_outputs", f"{label}_{alphabet_suffix}_int{interval}_results.npz"
)
sys.exit(0 if os.path.exists(fname) else 1)
EOF
}

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
            if already_complete "$provider" "$interval" "$alphabet" "$model" "$effort"; then
                echo "[$(date +%H:%M:%S)] [$provider] interval=$interval alphabet='$alphabet' -- already complete, skipping."
                continue
            fi
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
    logfile="logs/${provider}_finish.log"
    echo "Starting $provider (log: $logfile)..."
    run_provider "$provider" > "$logfile" 2>&1 &
    pids+=("$!")
    names+=("$provider")
done

echo "Launched ${#pids[@]} provider(s) in parallel: ${names[*]}"
echo "Tail any of them with: tail -f logs/<provider>_finish.log"

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[${names[$i]}] SUCCEEDED"
    else
        echo "[${names[$i]}] FAILED -- see logs/${names[$i]}_finish.log"
        status=1
    fi
done

exit "$status"
