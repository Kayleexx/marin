#!/usr/bin/env bash
# Baseline: Qwen3-1.7B-Base, bf16 weights, f32 math, greedy, batch 1, on CPU and CUDA.
#   scripts/bench_baseline.sh /path/to/Qwen3-1.7B-Base [out_dir]
# Close other heavy programs first: the CPU run needs about 4 GB of free RAM and timings are wall-clock.
set -euo pipefail
model="${1:?usage: bench_baseline.sh MODEL_DIR [OUT_DIR]}"
out="${2:-docs/results/baseline}"
mkdir -p "$out"
cargo build --release --locked -p cli
run=target/release/marin-run
for backend in cuda cpu; do
  for prompt in 32 128 512; do
    runs=5; warmup=1
    if [ "$backend" = cpu ]; then runs=3; fi
    if [ "$backend" = cpu ] && [ "$prompt" = 512 ]; then runs=2; warmup=0; fi
    name="$backend-p$prompt-t64"
    "$run" bench "$model" --backend "$backend" --prompt-tokens "$prompt" --tokens 64 \
      --runs "$runs" --warmup "$warmup" --profile --json "$out/$name.json" | tee "$out/$name.txt"
  done
done
