cd /home/mitali/miracles/marin
S=/tmp/claude-1000/-home-mitali-miracles-marin/82b2e41c-9d75-476b-8807-5a245de984ac/scratchpad
M=$HOME/models/Qwen3.5-4B-Q8; NEW=target/release/marin-run
cool(){ for i in $(seq 40); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader); [ "$t" -le 52 ] && break; sleep 5; done; }
fields(){ echo "$1" | awk '/first token/{t=$4} /^  prefill /{p=$2} /^  decode /{d=$2} /latency p50/{l=$4} END{printf "ttft %sms prefill %s decode %s p50 %sms", t,p,d,l}'; }
for i in 1 2 3; do
  cool; out=$($NEW bench $M --backend cuda --quantize int8 --prompt-tokens 128 --tokens 64 --runs 3 --warmup 1 2>&1); echo "no-sampling      : $(fields "$out")"
  cool; out=$($NEW bench $M --backend cuda --quantize int8 --prompt-tokens 128 --tokens 64 --runs 3 --warmup 1 --sample-gpu 2>&1); echo "--sample-gpu     : $(fields "$out")"
  cool; (nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader -lms 200 > $S/ext.$i.txt &) ; out=$($NEW bench $M --backend cuda --quantize int8 --prompt-tokens 128 --tokens 64 --runs 3 --warmup 1 2>&1); pkill -x nvidia-smi; echo "external poller  : $(fields "$out")  clk(max,mean MHz)=$(tr -d ' MHzW' < $S/ext.$i.txt | awk -F, '$1>500{n++;s+=$1; if($1>m)m=$1} END{printf "%d,%d",m,s/n}')"
done
echo SAMP_DONE
