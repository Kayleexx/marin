#!/bin/bash
# usage: chat_pressure.sh TARGET_FREE_MIB CTX TURNS PAUSE_AFTER REPEAT
cd /home/mitali/miracles/marin
S=/tmp/claude-1000/-home-mitali-miracles-marin/82b2e41c-9d75-476b-8807-5a245de984ac/scratchpad/v9
T=$1; CTX=$2; TURNS=$3; PA=$4; REP=${5:-60}
FLAG=$S/flag.$$; touch $FLAG; rm -f $FLAG.paused $FLAG.go
MODEL=$HOME/models/Qwen3.5-4B-Q8 CTX=$CTX TURNS=$TURNS PAUSE_AFTER=$PA FLAG=$FLAG REPEAT=$REP \
  ${ZZ:-target/release/deps/zz_long-f4938525ddcee533} --nocapture --test-threads 1 > $S/chat.$T.$CTX.out 2>&1 &
cp=$!
while [ ! -e $FLAG.paused ]; do sleep 1; kill -0 $cp 2>/dev/null || { echo chat died; exit 1; }; done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$T" != "none" ]; then
  mib=$((free - T))
  MIB=$mib FLAG=$FLAG target/release/deps/zz_ballast-545db35539485d0c --nocapture > $S/ballast.$T.out 2>&1 &
  bp=$!
  until grep -q "BALLAST held" $S/ballast.$T.out; do sleep 0.5; done
  echo "free before $free MiB, ballast $mib MiB, free now $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) MiB"
fi
touch $FLAG.go
while kill -0 $cp 2>/dev/null; do
  if [ -e $FLAG.failed ] && [ ! -e $FLAG.release ]; then
    rm -f $FLAG; [ -n "$bp" ] && wait $bp; bp=; sleep 1; touch $FLAG.release
  fi
  sleep 0.5
done
rm -f $FLAG
[ -n "$bp" ] && wait $bp
grep -E "LOADED|TURN|RETRY|panicked|error" $S/chat.$T.$CTX.out
