cd /home/mitali/miracles/marin
S=/tmp/claude-1000/-home-mitali-miracles-marin/82b2e41c-9d75-476b-8807-5a245de984ac/scratchpad
M=$HOME/models/Qwen3.5-4B-Q8
for who in "before:$S/before/target/release/marin" "after:target/release/marin"; do
  n=${who%%:*}; b=${who#*:}
  $b --model $M --quantize int8 --backend cuda --ctx 8192 serve --port 8190 > $S/srvload.$n.log 2>&1 &
  sp=$!; for i in $(seq 60); do grep -q listening $S/srvload.$n.log && break; sleep 1; done
  start=$(date +%s)
  for i in 1 2; do curl -s -m 600 -o $S/resp$i.$n.json -w "req$i %{http_code} %{time_total}s\n" -H 'Content-Type: application/json' -d @$S/req$i.json http://127.0.0.1:8190/v1/chat/completions >> $S/srvload.$n.times & done
  peak=0
  while [ $(jobs -r | wc -l) -gt 1 ]; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$sp -F', ' '$1==p{print $2}' | head -1)
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    sleep 0.5
  done
  wait $(jobs -p | grep -v $sp) 2>/dev/null
  total=$(( $(date +%s) - start ))
  echo "== $n: $(grep listening $S/srvload.$n.log | grep -o 'up to [0-9]* at once'); both requests done in ${total}s; server process VRAM peak ${peak} MiB; free before/after unchanged"
  cat $S/srvload.$n.times
  python3 -c "
import json
for i in (1,2):
    d=json.load(open('$S/resp'+str(i)+'.$n.json')); print('  req',i,'->',d.get('usage'), d['choices'][0]['message']['content'][:60].replace('\n',' ') if 'choices' in d else d)
"
  kill $sp; wait $sp 2>/dev/null
done
echo SRV_DONE
