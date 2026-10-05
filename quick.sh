#!/bin/bash
set -e

# test_gpt.py checkpoints progress after every trial, so on a transient
# provider error (e.g. a 504 Gateway Timeout) it's safe to just retry the
# same command -- it resumes from the last completed trial instead of
# starting over. Keep retrying until it exits 0, then (and only then) score
# the results; without this loop, a crash here would leave test_outputs
# stale and analyze_gpt.py would silently score old/incomplete data.
#
# Capped: not every failure is transient (e.g. a model consistently
# returning an empty final answer fails the same way every time), so give
# up after MAX_RETRY_ATTEMPTS rather than looping forever.
MAX_RETRY_ATTEMPTS=10
attempt=1
until python3 ./test_gpt.py --provider qwen --alphabet "z y x w v u t s r q p o n m l k j i h g f e d c b a" --interval_size 2 --trials 100 --effort low; do
    if (( attempt >= MAX_RETRY_ATTEMPTS )); then
        echo "test_gpt.py failed $MAX_RETRY_ATTEMPTS times in a row -- giving up (this looks like a persistent failure, not a transient one)."
        exit 1
    fi
    echo "test_gpt.py failed (attempt $attempt/$MAX_RETRY_ATTEMPTS) -- retrying (progress is checkpointed, so this resumes rather than restarting) in 10s..."
    sleep 10
    attempt=$((attempt + 1))
done

python3 ./analyze_gpt.py --provider qwen --gpt_engine "Qwen/Qwen3.8-27B:deepinfra" --alphabet "z y x w v u t s r q p o n m l k j i h g f e d c b a" --interval_size 2 --effort low