#!/usr/bin/env bash
# Builds the prompt folders read by `marin-run guess --prompts DIR`: one folder per kind of work, one .txt per message.
# Code and documents come from this repository, so the prompts look like the work Marin will be asked to do.
set -euo pipefail

out=${1:?usage: guess_prompts.sh OUTPUT_DIR}
cd "$(dirname "$0")/.."
rm -rf "$out"
mkdir -p "$out"/{chat,chat-doc,code-write,code-edit,code-explain,long}

say() { # say CLASS NAME TEXT
  printf '%s\n' "$3" >"$out/$1/$2.txt"
}

with_file() { # with_file CLASS NAME INSTRUCTION FILE [MAX_BYTES] [FENCE]
  {
    printf '%s\n\n' "$3"
    printf '```%s\n' "${6:-rust}"
    head -c "${5:-100000}" "$4"
    printf '\n```\n'
  } >"$out/$1/$2.txt"
}

say chat mutex "Explain what a mutex is and when I should use one."
say chat process-thread "What's the difference between a process and a thread?"
say chat lisbon "Give me a 3-day itinerary for a first trip to Lisbon."
say chat email "Write a polite email asking my landlord to fix a leaking tap in my kitchen."
say chat hashmap "Explain how a hash map works and why lookups are usually O(1)."
say chat sleep "What are good habits for sleeping better? Keep it practical."
say chat tcp-udp "Compare TCP and UDP, with an example use for each."
say chat meal-prep "Help me plan a weekly meal prep for a vegetarian on a budget."
say chat romeo "Summarize the plot of Romeo and Juliet in a few paragraphs."
say chat public-key "How does public-key cryptography work, in simple terms?"
say chat sky "Why is the sky blue, and why are sunsets red?"
say chat interview "Give me tips for preparing for a software engineering interview."

with_file chat-doc readme-summary "Summarize this README in five bullet points." README.md 100000 markdown
with_file chat-doc readme-table "List every crate named in this README and what it does, as a table." README.md 100000 markdown
with_file chat-doc bench-quant "What does this document say about int8 versus int4 weights? Answer in a short paragraph." BENCHMARKS.md 12000 markdown
if [ -f marin_spec_v0.4.md ]; then
  sed -n 1122,1330p marin_spec_v0.4.md >"$out/.spec-devices"
  with_file chat-doc spec-devices "Summarize the design of this multi-device section in plain words." "$out/.spec-devices" 100000 markdown
  sed -n 1611,1760p marin_spec_v0.4.md >"$out/.spec-proactive"
  with_file chat-doc spec-checklist "Turn these notes into a checklist of tasks, grouped by topic." "$out/.spec-proactive" 100000 markdown
  sed -n 1426,1610p marin_spec_v0.4.md >"$out/.spec-context"
  with_file chat-doc spec-faq "Write a short FAQ with five questions and answers based on this text." "$out/.spec-context" 100000 markdown
else
  echo "skipping the prompts built from marin_spec_v0.4.md: it is not tracked and not in this checkout" >&2
fi

say code-write semver "Write a Rust function that parses a semantic version string like 1.2.3-beta into a struct, with error handling and unit tests."
say code-write csv-average "Write a Python script that reads a CSV file and prints the average of each numeric column."
say code-write backup "Write a bash script that backs up a directory into a timestamped tar.gz and keeps only the last 5 backups."
say code-write lru "Implement an LRU cache in Rust with get and put in O(1) time."
say code-write debounce "Write a debounce function in TypeScript with proper types, and show an example of using it."
say code-write sql-top "Write SQL to find the top 3 customers by total spend in each month, given tables customers(id, name) and orders(id, customer_id, amount, created_at)."
say code-write intervals "Write a Python function that merges overlapping intervals, with a docstring and tests."
say code-write go-server "Write a Go HTTP server with a /health endpoint and graceful shutdown on SIGINT."

with_file code-edit kv-truncate "Add a \`truncate(&mut self, len: usize)\` method to KvCache that shrinks the committed length and returns an error if len is larger than the current length. Show the complete updated file." crates/kv/src/lib.rs
with_file code-edit sampler-docs "Add a doc comment to every public item in this file. Return the complete file." crates/sampler/src/lib.rs
with_file code-edit ops-clamp "Add a function \`clamp_in_place(x: &mut [f32], low: f32, high: f32)\` next to the others, in the same style, plus a unit test. Show the complete updated file." crates/tensor/src/ops.rs
with_file code-edit conv-last "Add a method \`last_user_message(&self) -> Option<&str>\` to Conversation and a unit test for it. Show the complete updated file." crates/chat/src/conversation.rs
with_file code-edit safetensors-names "Change \`read_bf16\` so its error always includes the tensor name and the file path. Show the complete updated file." crates/loader/src/safetensors.rs
with_file code-edit generate-timing "Change \`generate\` so the returned Outcome also reports the total time spent inside model.forward calls. Show the complete updated file." crates/runtime/src/generate.rs

with_file code-explain kv "Explain what this code does, step by step." crates/kv/src/lib.rs
with_file code-explain ops "Review this code for bugs and style problems and list what you would change." crates/tensor/src/ops.rs
with_file code-explain conversation "Explain how this code decides what to drop when the conversation is too long." crates/chat/src/conversation.rs
with_file code-explain sampler "Explain how the sampler picks the next token, including temperature, top-k and top-p." crates/sampler/src/lib.rs

with_file long bench-findings "Summarize the main findings of this benchmark document in eight bullet points, keeping the numbers." BENCHMARKS.md 10500 markdown
if [ -f marin_spec_v0.4.md ]; then
  sed -n 1,640p marin_spec_v0.4.md >"$out/.spec-head"
  with_file long spec-overview "Write a one-page overview of this project for a new contributor." "$out/.spec-head" 12000 markdown
fi
with_file long verify-explain "Explain what this file does and list possible improvements." crates/cli/src/verify.rs
with_file long generate-review "Write documentation for this file in markdown: what it does, how to use it, and its edge cases." crates/runtime/src/generate.rs

rm -f "$out"/.spec-*
echo "wrote $(find "$out" -name '*.txt' | wc -l) prompts to $out"
