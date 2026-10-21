#!/usr/bin/env bash
# Builds the scenario folders read by `marin-run spec-bench --prompts DIR`: whole-file edits, diff-style edits, fresh
# code and chat, each with a short context and a long one (about 4k tokens). Code and documents come from this repo.
set -euo pipefail

out=${1:?usage: spec_prompts.sh OUTPUT_DIR}
cd "$(dirname "$0")/.."
rm -rf "$out"
mkdir -p "$out"/{edit-file-short,edit-file-long,edit-diff-short,edit-diff-long,fresh-short,fresh-long,chat-short,chat-long}

say() { printf '%s\n' "$3" >"$out/$1/$2.txt"; }

# with_files SCENARIO NAME INSTRUCTION FILE...: the instruction, then each file in a fenced block
with_files() {
  local scenario=$1 name=$2 instruction=$3
  shift 3
  {
    printf '%s\n' "$instruction"
    for file in "$@"; do
      printf '\n// %s\n```rust\n' "$file"
      cat "$file"
      printf '\n```\n'
    done
  } >"$out/$scenario/$name.txt"
}

diff_note="Reply with only a unified diff of the change. Do not repeat unchanged code."
whole_note="Show the complete updated file."

short_a=crates/runtime/src/lib.rs
short_b=crates/model/src/qwen3/workspace.rs
long_a=(crates/cli/src/verify.rs crates/model/src/qwen3/workspace.rs)
long_b=(crates/sampler/src/lib.rs crates/chat/src/conversation.rs)

ask_a="Add a method \`remaining(&self) -> usize\` to the SequenceCache trait, with a default implementation using capacity and used."
ask_b="Rename the field \`last\` to \`final_hidden\` everywhere in this file."
ask_c="Add a doc comment to every function in the first file."
ask_d="Rename \`drop_oldest_exchange\` to \`drop_oldest_turn\` and update every use."

with_files edit-file-short runtime "$ask_a $whole_note" $short_a
with_files edit-file-short workspace "$ask_b $whole_note" $short_b
with_files edit-diff-short runtime "$ask_a $diff_note" $short_a
with_files edit-diff-short workspace "$ask_b $diff_note" $short_b
with_files edit-file-long verify-docs "$ask_c Show the complete updated first file." "${long_a[@]}"
with_files edit-file-long rename-turn "$ask_d Show the complete updated conversation.rs." "${long_b[@]}"
with_files edit-diff-long verify-docs "$ask_c $diff_note" "${long_a[@]}"
with_files edit-diff-long rename-turn "$ask_d $diff_note" "${long_b[@]}"

say fresh-short semver "Write a Rust function that parses a semantic version string like 1.2.3-beta into a struct, with error handling and unit tests."
say fresh-short lru "Implement an LRU cache in Rust with get and put in O(1) time."
with_files fresh-long cosine "Following the style of the code below, write a new function \`cosine_similarity(a: &[f32], b: &[f32])\` that returns an error for mismatched or empty input, with unit tests." "${long_a[@]}"
with_files fresh-long ringbuf "Following the style of the code below, write a small fixed-size ring buffer of f32 values with push, iter and mean, with unit tests." "${long_b[@]}"

say chat-short mutex "Explain what a mutex is and when I should use one."
say chat-short tcp-udp "Compare TCP and UDP, with an example use for each."
if [ -f marin_spec_v0.4.md ]; then
  for name in risks goals; do
    case $name in
      risks) question="What are the main risks or open questions in this document? Answer with a short list." ;;
      goals) question="Summarize the goals of this document in five bullet points." ;;
    esac
    {
      printf '%s\n\n```markdown\n' "$question"
      head -c 15000 marin_spec_v0.4.md
      printf '\n```\n'
    } >"$out/chat-long/$name.txt"
  done
else
  echo "skipping the prompts built from marin_spec_v0.4.md: it is not tracked and not in this checkout" >&2
fi
echo "wrote $(find "$out" -name '*.txt' | wc -l) prompts to $out"
