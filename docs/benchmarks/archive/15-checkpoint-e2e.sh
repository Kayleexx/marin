#!/usr/bin/env bash
# Resume of a long Qwen3.5-4B-Q8 chat with and without a checkpoint. Usage: 15-checkpoint-e2e.sh <essays 1-8> [marin binary]
# Builds a conversation of <essays> answers of about 420 tokens, saves and checkpoints it, then times a new process
# that resumes it (once from the text alone, once from the checkpoint) and asks one question.
set -euo pipefail
essays=${1:?number of essays}
marin=${2:-target/release/marin}
model=${MARIN_QWEN35_Q8_DIR:-$HOME/models/Qwen3.5-4B-Q8}
home=$(mktemp -d)
run() { MARIN_HOME=$home NO_COLOR=1 "$marin" --model "$model" --quantize int8 --backend cuda --temp 0 --ctx 4096 --max-tokens "$1"; }
ms() { echo $(( ($(date +%s%N) - $1) / 1000000 )); }
topics=("the printing press" "the steam engine" "the telegraph" "the compass" "the telescope" "the microscope" "the railway" "the telephone")
script=""
for i in $(seq 0 $((essays - 1))); do script+="Write a detailed essay of about 350 words on ${topics[$i]}."$'\n'; done
printf '%s/save long\n/checkpoint long\n' "$script" | run 420 | grep -E "^kept the model|^error:" || true
python3 - "$home/sessions/long.json" "$home/sessions/plain.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["name"] = "plain"; json.dump(d, open(sys.argv[2], "w"), separators=(",", ":"))
PY
for name in plain long; do
  t=$(date +%s%N)
  printf '/resume %s\nIn three words, what was the first essay about?\n' "$name" | run 20 > "$home/$name.out"
  echo "resume $name + one question: $(ms "$t") ms"
  grep -hE "loaded the model|ignored" "$home/$name.out" || true
done
cmp <(tail -n 1 "$home/plain.out") <(tail -n 1 "$home/long.out") && echo "ANSWERS IDENTICAL"
rm -rf "$home"
