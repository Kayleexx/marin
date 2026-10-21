cd /home/mitali/miracles/marin
S=/tmp/claude-1000/-home-mitali-miracles-marin/82b2e41c-9d75-476b-8807-5a245de984ac/scratchpad
M=$HOME/models/Qwen3.5-4B-Q8
OLD=$S/before/target/release/marin-run; NEW=target/release/marin-run
cool(){ for i in $(seq 60); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader); [ "$t" -le 52 ] && break; sleep 5; done; }
one(){ # label binary prompt tokens runs warmup
  cool
  $2 bench $M --backend cuda --quantize int8 --prompt-tokens $3 --tokens $4 --runs $5 --warmup $6 --sample-gpu > $S/ablong.tmp 2>&1 &
  pid=$!; peak=0
  while kill -0 $pid 2>/dev/null; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$pid -F', ' '$1==p{print $2}' | head -1)
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    sleep 0.4
  done
  out=$(cat $S/ablong.tmp)
  ttft=$(echo "$out" | grep "first token" | awk '{print $4}'); pf=$(echo "$out" | grep -E "^  prefill " | awk '{print $2}'); dc=$(echo "$out" | grep -E "^  decode " | awk '{print $2}'); lat=$(echo "$out" | grep "latency p50" | awk '{print $4}')
  echo "$1 prompt $3 tok $4 | ttft ${ttft}ms prefill $pf decode $dc p50 ${lat}ms | process VRAM peak ${peak} MiB"
}
for pair in 1 2 3; do one BEFORE $OLD 128 64 3 1; one AFTER $NEW 128 64 3 1; done
one BEFORE $OLD 2048 33 1 0; one AFTER $NEW 2048 33 1 0
one BEFORE $OLD 8192 33 1 0; one AFTER $NEW 8192 33 1 0
one BEFORE $OLD 14336 33 1 0; one AFTER $NEW 14336 33 1 0
echo ABLONG_DONE
