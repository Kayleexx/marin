cd /home/mitali/miracles/marin
S=/tmp/claude-1000/-home-mitali-miracles-marin/82b2e41c-9d75-476b-8807-5a245de984ac/scratchpad
M=$HOME/models/Qwen3.5-4B-Q8
OLD=$S/base/target/release/marin-run; NEW=target/release/marin-run
state(){ nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader | tr -d ' '; }
cool(){ for i in $(seq 40); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader); [ "$t" -le 52 ] && break; sleep 5; done; }
one(){ # label binary prompt tokens runs warmup
  cool; pre=$(state)
  out=$($2 bench $M --backend cuda --quantize int8 --prompt-tokens $3 --tokens $4 --runs $5 --warmup $6 --sample-gpu 2>&1)
  ttft=$(echo "$out" | grep "first token" | awk '{print $4}'); pf=$(echo "$out" | grep -E "^  prefill " | awk '{print $2}'); dc=$(echo "$out" | grep -E "^  decode " | awk '{print $2}'); lat=$(echo "$out" | grep "latency p50" | awk '{print $4}')
  clk=$(echo "$out" | grep -iE "clock|power cap|mean" | tr -s ' ' | tr '\n' '|' | cut -c1-200)
  echo "$1 p$3 t$4 | ttft ${ttft}ms prefill $pf decode $dc p50 ${lat}ms | start(temp,clk,W)=$pre | $clk"
}
for pair in 1 2 3; do
  one OLD $OLD 512 4 1 0; one NEW $NEW 512 4 1 0
done
for pair in 1 2; do
  one OLD $OLD 128 64 3 1; one NEW $NEW 128 64 3 1
done
one OLD $OLD 2048 4 1 0; one NEW $NEW 2048 4 1 0
echo AB_DONE
