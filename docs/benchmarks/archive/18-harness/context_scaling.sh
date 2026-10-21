cd /home/mitali/miracles/marin
S=/tmp/claude-1000/-home-mitali-miracles-marin/82b2e41c-9d75-476b-8807-5a245de984ac/scratchpad
M=$HOME/models/Qwen3.5-4B-Q8
cool(){ for i in $(seq 60); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader); [ "$t" -le 52 ] && break; sleep 5; done; }
for n in 128 2048 8192 14336; do
  cool
  out=$(timeout 500 target/release/marin-run bench $M --backend cuda --quantize int8 --prompt-tokens $n --tokens 17 --runs 1 --warmup 0 --profile 2>&1)
  echo "=== prompt $n"
  echo "$out" | grep -E "first token|^  prefill |^  decode |latency p50|gpu memory in use \(peak|kv cache"
  echo "$out" | grep -E "state transaction copy" 
  echo "$out" | grep -E "one-token passes|decode:|^  decode: " | head -2
done
echo CTX_DONE
