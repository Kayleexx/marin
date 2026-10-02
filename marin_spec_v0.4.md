# Marin

## A Hardware-Adaptive Local Inference Runtime

**Status:** v0.4 spec  
**Primary goal:** learn inference engineering deeply by building and optimizing a real local inference runtime for constrained consumer hardware.

---

# 1. What is Marin?

Marin is a private, local-first intelligence layer built on top of a hardware-adaptive inference runtime.

Its technical core is still inference engineering.

Its product surface eventually becomes a small personal assistant that can:

```text
do things when asked

help with coding

remember ongoing work

understand what is happening on the device

surface useful suggestions

stay quiet when nothing useful needs to be said

follow the same session across laptop and phone
```

Marin is **not primarily an AI agent framework**.

The core project is still:

> build a small inference engine, understand every major part of the inference path, then progressively optimize it for weak and consumer-grade machines.

The assistant layer exists to give the runtime a demanding real-world workload.

The long-term product thesis is:

> Marin understands what the user is doing, remembers enough to continue it, can act when asked, and only interrupts when it has something genuinely useful to contribute, while keeping useful intelligence local and minimizing compute, context, and data collection.

Marin should eventually have three user-facing modes:

```text
ASSISTANT MODE
"do this for me"

CODING MODE
replace a meaningful amount of Codex / Claude Code usage

PROACTIVE MODE
"this may help you"
```

All three modes share the same:

```text
session state

memory

context engine

tool system

inference runtime

hardware-aware routing
```

The main engineering question remains:

> how much useful intelligence can we squeeze out of a fixed amount of compute?

The main product question becomes:

> how useful can a private local assistant become while collecting as little data, using as little context, consuming as little compute, and causing as few interruptions as possible?

# 2. Why Marin exists

A lot of local AI projects sit above inference engines.

They call:

```text
llama.cpp
vLLM
SGLang
Ollama
MLC
```

and focus mainly on:

```text
agents
tools
memory
UI
orchestration
```

Those are useful products, but they hide most of the actual inference work.

Marin exists to go lower.

We want to directly understand and implement things such as:

```text
model loading
tensor layout
prefill
decode
attention
KV cache
quantization
memory movement
batching
scheduling
speculative decoding
kernel behavior
hardware adaptation
state persistence
offloading
context limits
```

The agent layer comes later.

---

# 3. North-star goal

Marin should eventually behave like:

```bash
marin serve model/
```

and automatically adapt itself to the machine.

Example:

```text
hardware

RTX 5050 Laptop
8 GB VRAM
16 GB RAM

selected runtime plan

model          Qwen3.5-9B Q4
GPU placement  maximum
context        12K
KV             Q8
KV tiering     GPU → RAM
speculation    enabled
draft model    0.8B
scheduler      interactive
```

On weaker hardware:

```text
hardware

integrated GPU
8 GB RAM

selected runtime plan

model          2B Q4
GPU placement  partial
context        4K
KV             Q4
speculation    disabled
scheduler      memory-first
```

The same runtime should scale itself down gracefully.

---

# 4. Baseline hardware

Primary development machine:

```text
GPU: RTX 5050 Laptop
VRAM: 8 GB GDDR7
CPU: Ryzen 7
RAM: 16 GB
OS: Arch Linux
```

Marin must never assume this exact machine.

Eventually Marin should detect:

```text
CPU
RAM
GPU
VRAM
GPU architecture
supported instruction sets
available disk
memory bandwidth where measurable
supported compute backend
power mode
```

and construct a hardware profile.

---

# 5. Project philosophy

Marin follows a few strict rules.

## 5.1 Correctness before optimization

Every optimization starts from a correct baseline.

Do not optimize something we cannot validate.

## 5.2 Measure everything

Never accept:

> "it feels faster."

Every change gets:

```text
before
vs
after
```

## 5.3 Real workloads matter

Synthetic microbenchmarks are useful.

They are not enough.

Every optimization should eventually be tested against real tasks.

## 5.4 Keep the first implementation simple

Do not begin with:

```text
distributed inference
multi-GPU
MoE
giant models
custom PTX for everything
complicated hybrid architectures
```

Start with a normal decoder-only transformer.

## 5.5 Build upward

Marin should progress like:

```text
correct inference
        ↓
measured inference
        ↓
efficient inference
        ↓
adaptive inference
        ↓
real applications
```

## 5.6 No feature slop

Every new feature must satisfy at least one of these:

```text
1. makes Marin materially more useful

or

2. teaches us something meaningful about inference / systems
```

If it does neither, do not build it.

Avoid features added only because other assistants have them.

## 5.7 Minimize four things

Marin should aggressively minimize:

```text
compute used

context used

private data collected

interruptions made
```

while maximizing:

```text
usefulness
```

This should influence both runtime architecture and product behavior.

---

# 6. Model strategy

## 6.1 First model

Do not begin the runtime implementation with Qwen3.5.

Qwen3.5 uses a hybrid architecture, which adds unnecessary complexity while learning the basics.

Start with a small conventional decoder-only transformer.

Requirements:

```text
normal attention
RMSNorm or LayerNorm
RoPE
standard MLP
simple tokenizer
available safetensors weights
small enough to debug locally
```

The exact first model can be chosen when implementation begins.

The first goal is correctness, not capability.

## 6.2 Later model family

Once the runtime works correctly, move toward models useful for the real workload.

Likely progression:

```text
tiny conventional model
        ↓
1B–3B model
        ↓
4B model
        ↓
Qwen3.5-9B or equivalent
        ↓
larger models when optimizations permit
```

## 6.3 Qwen3.5 role

Qwen3.5 remains useful later for:

```text
real coding workload
tool use
speculative decoding experiments
routing
hybrid architecture experiments
larger context
consumer-GPU optimization
```

But it should not be the first architecture we implement.

---

# 7. Core runtime architecture

Initial architecture:

```text
marin-run
    local inference CLI

marin-core
    model execution

marin-tensor
    tensor representation / layouts

marin-model
    architecture definitions

marin-kv
    KV cache

marin-sampler
    decoding / sampling

marin-loader
    weights + tokenizer loading

marin-metrics
    instrumentation

marin-hw
    hardware detection
```

Later:

```text
marin-scheduler
marin-quant
marin-speculate
marin-memory-manager
marin-server
```

The project should remain modular enough that different backends can be compared.

---

# 8. Execution backends

Marin should not try to write every kernel from scratch on day one.

Initial strategy:

```text
Rust runtime logic
+
existing BLAS / CUDA primitives where useful
```

Possible backend progression:

```text
CPU baseline
        ↓
CUDA backend
        ↓
custom kernels for important hot paths
        ↓
Vulkan / Metal / other platforms
```

The goal is to understand bottlenecks first.

Only replace kernels when there is evidence that doing so matters.

---

# 9. V0 — correct inference

## Goal

Make one small transformer generate correct text.

No major optimization yet.

## Implement

```text
safetensors loader
tokenizer integration
embedding lookup
RMSNorm / LayerNorm
RoPE
Q/K/V projections
self-attention
causal masking
MLP
residual connections
final norm
LM head
sampling
autoregressive decode
```

## Basic KV cache

Implement the simplest correct KV cache.

Initial layout can be contiguous.

No paging yet.

## CLI

Example:

```bash
marin-run ./model
```

```text
> hello

hello! ...
```

## V0 completion condition

Marin output should numerically match a trusted reference implementation closely enough for the same model and prompt.

Validation should include:

```text
logit comparisons
token output comparisons
layer-level debugging where necessary
```

---

# 10. V1 — observability and benchmark harness

Before serious optimization, expose the inference path.

## Request metrics

```text
TTFT
prefill latency
prefill tokens/sec
decode tokens/sec
inter-token latency
p50 ITL
p95 ITL
total generation time
```

## Layer metrics

Measure:

```text
embedding
norm
QKV projection
attention
RoPE
MLP
LM head
```

Example trace:

```text
token 42

embedding      0.03 ms

layer 0
  norm         0.01 ms
  qkv          0.29 ms
  rope         0.02 ms
  attention    0.16 ms
  mlp          0.41 ms

...

lm_head        0.28 ms
```

## Resource metrics

```text
VRAM usage
RAM usage
GPU utilization
CPU utilization
memory copies
PCIe transfer where measurable
disk IO
kernel time
memory bandwidth
power where measurable
```

## Benchmark harness

Every benchmark should store:

```text
model
precision
quantization
context length
batch size
prompt length
output length
hardware
runtime version
results
```

---

# 11. V2 — model execution optimization

Before quantization, understand baseline execution.

Experiment with:

```text
tensor layout
contiguous vs packed layouts
operation fusion
kernel launch overhead
memory copies
buffer reuse
allocation reuse
CUDA streams
prefill/decode specialization
```

Questions:

```text
what is compute-bound?
what is memory-bound?
which ops dominate prefill?
which ops dominate decode?
where are synchronization points?
where are unnecessary copies?
```

This phase is about building intuition for the real hot path.

---

# 12. V3 — quantization

Now attack model memory and bandwidth.

## Controlled precision ladder

Test:

```text
FP32 where useful for debugging
BF16
FP16
INT8
Q8
Q6
Q5
Q4
Q3
lower-bit / mixed schemes
```

## Eventually implement

At least some quantization logic ourselves.

Possible progression:

```text
weight-only INT8
        ↓
grouped INT8
        ↓
INT4 / Q4
        ↓
different group sizes
        ↓
mixed precision by layer
        ↓
hardware-specific formats
```

## Measure

```text
model size
VRAM
RAM
TTFT
prefill throughput
decode throughput
dequantization overhead
quality
task success
power
```

## Main questions

```text
when does lower-bit quantization stop helping?
when does dequantization become expensive?
which layers are most sensitive?
should some layers remain higher precision?
how much more context fits after quantization?
when does full-GPU placement beat higher precision with CPU offload?
```

---

# 13. V4 — KV cache architecture

Now make inference-state memory a first-class subsystem.

## Start

```text
contiguous KV cache
```

Then evolve toward:

```text
paged KV cache
```

## Goals

Support:

```text
allocation
freeing
reuse
fragmentation control
page tracking
sequence ownership
prefix sharing
```

## KV precision

Experiment with:

```text
FP16
BF16
Q8
Q4
```

Measure:

```text
memory saved
quality change
decode latency
maximum usable context
```

---

# 14. V5 — tiered KV memory

Turn KV into a memory hierarchy.

```text
VRAM
fastest
    ↓
pinned system RAM
    ↓
NVMe SSD
slowest
```

## Implement

```text
KV page eviction
KV page promotion
KV page demotion
async copies
prefetching
cache residency tracking
hot/cold classification
```

## Initial eviction policies

Compare:

```text
LRU
LFU
recency + reuse score
prefix-aware eviction
cost-aware eviction
```

## Questions

```text
when is RAM offload worth it?
when does PCIe become the bottleneck?
when is SSD useful at all?
how far ahead must we prefetch?
can we predict reuse better than LRU?
which pages should remain permanently hot?
```

This is one of Marin's most important systems areas.

---

# 15. V6 — prefix reuse and persistent inference state

Avoid recomputing work that already happened.

## Prefix caching

Support:

```text
shared system prompts
shared repository context
shared tool definitions
repeated long prefixes
```

## Persistent state

Investigate:

```text
saving reusable inference state
restoring cached prefixes
worker restart recovery
cache snapshots
state fingerprints
```

Goal:

```text
process dies
        ↓
runtime restarts
        ↓
use persistent state when possible
        ↓
avoid full replay
```

---

# 16. V7 — batching and scheduler

Now support multiple concurrent requests.

## Implement

```text
request queue
continuous batching
prefill batching
decode batching
request priority
preemption
admission control
```

## Chunked prefill

Large prompts should not completely block interactive requests.

Experiment with:

```text
prefill chunk size
decode interleaving
interactive priority
background throughput
```

## Scheduler inputs

Possible signals:

```text
request age
prompt length
remaining output
memory pressure
cache locality
interactive vs background
latency target
```

## Goal

Make one consumer machine feel responsive while several requests are active.

---

# 17. V8 — speculative decoding

Implement speculative decoding directly.

## Basic design

```text
draft model
    ↓
propose N tokens

target model
    ↓
verify

accepted tokens
    ↓
commit

rejected token
    ↓
correct + continue
```

## Experiments

```text
fixed draft length
dynamic draft length
0.5B draft
0.8B draft
2B draft
n-gram speculation
suffix-based speculation
MTP where available
```

## Metrics

```text
acceptance rate
draft cost
verification cost
tokens accepted per verification step
ITL
TPS
VRAM overhead
total memory
power
```

## Important goal

Do not assume speculation always helps.

Build a controller that can decide:

```text
speculate
or
decode normally
```

based on live conditions.

---

# 18. V9 — adaptive runtime

Now Marin becomes hardware-aware.

## Hardware planner

Given:

```text
model
VRAM
RAM
CPU
GPU
context target
```

produce an execution plan.

Possible decisions:

```text
quantization
GPU placement
CPU placement
KV precision
KV tiering
context limit
batch size
speculative decoding
draft model
scheduler policy
```

## Dynamic adaptation

Later, modify plans while running.

Example:

```text
VRAM pressure increases
        ↓
quantize / evict KV
        ↓
move cold KV to RAM
        ↓
reduce background batch
        ↓
keep interactive request alive
```

Marin should optimize for useful behavior, not a static benchmark.

---

# 19. V10 — model routing

Once Marin can host multiple models:

```text
0.8B
2B
4B
9B
```

route requests based on:

```text
task complexity
latency target
available memory
context length
quality requirement
power mode
```

Sometimes the best inference optimization is:

```text
use a smaller model
```

Sometimes:

```text
use no model
```

The runtime should expose this capability to higher layers.

---

# 20. V11 — continuity, hibernation, migration

This is the later systems-heavy phase.

Explore:

```text
worker crash recovery
model hibernation
lazy model loading
hot model swapping
warm standby workers
GPU → CPU migration
worker → worker migration
machine → machine migration
session checkpointing
partial inference-state restoration
```

Long-term idea:

> inference workers are disposable. useful session state survives.

---

# 21. Low-end hardware as a first-class target

Marin should continuously test under artificial hardware limits.

Profiles:

```text
8 GB VRAM
6 GB VRAM
4 GB VRAM
2 GB VRAM
16 GB RAM
8 GB RAM
CPU-only
slow SSD
```

Every new feature should answer:

```text
does this help weak hardware?
does this hurt weak hardware?
where is the crossover point?
```

---


# 21.1 Multi-device inference as a learning target

The phone should be part of Marin's architecture because it gives us a second, very different hardware target.

The principle is:

> Marin follows the user across devices. The model does not have to.

Marin should separate:

```text
SESSION
portable logical state

NODE
device capabilities

RUNTIME
model execution

CLIENT
how the user interacts with Marin
```

## Phase A — phone as client

The first mobile version should stay simple.

```text
phone client
      ↓
secure connection
      ↓
marin server on laptop
      ↓
RTX 5050 inference
```

The phone does not run the main model yet.

It should be able to:

```text
open existing sessions
send prompts
stream tokens
see tool activity
cancel work
inspect diffs
resume coding tasks
```

The terminal does not need to remain open.

`marin-server` or the Marin daemon owns the running session.

## Phase B — phone as local fallback node

Later the phone becomes a real Marin node.

Example:

```text
PHONE NODE

small local model
limited context
mobile GPU / NPU / CPU
restricted tools
battery-sensitive runtime
```

while the laptop remains:

```text
LAPTOP NODE

larger model
CUDA
filesystem
shell
git
compiler
large context
```

Marin should discover each node's capabilities.

Possible node descriptor:

```text
node_id
device type
CPU
RAM
GPU / NPU
available accelerator memory
supported backends
available models
tool capabilities
battery state
network state
```

## Capability-aware routing

A request should not always execute on the same device.

Example:

```text
"what did we decide about KV paging?"
        ↓
phone can probably answer locally
```

but:

```text
"fix the failing Rust tests in ~/src/marin"
        ↓
requires repo + shell + compiler + stronger model
        ↓
route to laptop
```

Routing should eventually consider:

```text
task complexity
required tools
model capability
available context
memory pressure
latency
network cost
battery
device availability
```

## Offline behavior

If the laptop is unavailable:

```text
laptop offline
      ↓
phone-local fallback
```

The user should still get reduced functionality rather than Marin becoming completely unavailable.

Example:

```text
running locally on phone

model: 2B Q4
context: 4K

unavailable capabilities:
- desktop repo access
- CUDA workloads
- large-model reasoning
```

When the laptop returns, stronger tasks can route back to it.

## Portable state vs device-local state

Marin must explicitly separate these.

### Portable state

```text
conversation
task state
agent memory
summaries
tool history
workspace metadata
plans
artifacts
```

### Device-local inference state

```text
KV cache
GPU buffers
loaded weights
CUDA graphs
prefix cache
runtime allocations
background processes
```

Portable state can move between nodes.

Device-local acceleration state may need to be reconstructed.

This creates a useful inference engineering problem:

> how little state needs to move before another device can continue useful inference?

## Session migration

Long term:

```text
laptop
   ↓
phone
   ↓
laptop
```

should preserve one logical session.

Initially migration means:

```text
transfer portable state
        ↓
reconstruct inference state
```

Later experiments can investigate:

```text
prefix-state transfer
cache-state transfer
compatible KV transfer
partial checkpoint transfer
cross-device prefetch
```

Do not assume raw KV migration will always be worthwhile.

Measure transfer size and reconstruction cost.

## Heterogeneous model routing

Different nodes may run different model sizes.

Example:

```text
phone
0.8B / 2B

laptop
4B / 9B
```

Marin should eventually answer:

```text
can the small model handle this locally?
should the task escalate?
is network transfer more expensive than local inference?
should context be compressed before transfer?
should the laptop prefill while the phone keeps the UI responsive?
```

These are real experiments, not just product features.

## Multi-device benchmark suite

Add benchmarks for:

```text
local phone inference
remote laptop inference
phone → laptop escalation
session handoff
context transfer
reconstruction latency
network interruption
laptop reconnect
battery-sensitive routing
```

Metrics include:

```text
end-to-end latency
TTFT
tokens/sec
bytes transferred
reconstruction time
energy / battery cost where measurable
task success
escalation rate
```

## Why this belongs in Marin

The phone is useful because it forces Marin to handle hardware that is dramatically weaker and architecturally different from the development laptop.

That directly supports the core goal:

> learn how inference systems adapt across constrained hardware rather than optimizing only one GPU.

# 21.2 Situational context engine

Marin should eventually understand enough about the user's current device state to be useful without continuously recording everything.

The design principle is:

> situational awareness, not surveillance.

Prefer structured events over raw screenshots whenever possible.

Initial context sources should focus on development work:

```text
active application
active window
current repository
active file
filesystem changes
git state
terminal commands
terminal exit codes
compiler / test failures
Marin conversation state
selected text when explicitly exposed
```

Possible later opt-in sources:

```text
browser URL / title
clipboard
notifications
calendar
phone events
screenshots only when needed
camera / microphone only as explicit future capabilities
```

Do not start with:

```text
24/7 screen recording
24/7 microphone capture
continuous OCR
continuous vision inference
whole-life indexing
```

## Event-driven context

Marin should build a compact current-state model from device events.

Example:

```text
editor event:
active_file = crates/kv/src/page.rs

terminal event:
command = cargo test
exit = 101

git event:
modified = crates/kv/src/page.rs

recent activity:
same task for 34 minutes
```

Marin may derive:

```text
current situation

project:
marin

task:
debugging KV allocator

active file:
crates/kv/src/page.rs

recent event:
cargo test failed

confidence:
high
```

This state should be much smaller than dumping raw device history into the LLM.

## Context TTL

Not all context should live forever.

Each context item should have a scope and lifetime.

Examples:

```text
clipboard
TTL: seconds

terminal failure
TTL: minutes / current task

active file
TTL: current session

project build command
persistent project memory

architecture decision
persistent project memory
```

Default behavior:

> expire aggressively unless there is a clear reason to retain something.

## Sensitive-source filtering

Before an event enters Marin's working context:

```text
raw event
   ↓
source policy
   ↓
sensitive?
   ├── yes → discard / redact
   └── no
        ↓
semantic extraction
        ↓
ephemeral context
```

Marin should support source-level blocking for things such as:

```text
password managers
private browser windows
selected apps
selected directories
selected notifications
```

Privacy should come from data minimization, not merely storing unlimited data locally.

## Context inspection

The user should always be able to inspect what Marin currently knows.

Example:

```bash
marin context
```

Possible output:

```text
NOW

app       zed
project   ~/dev/marin
file      crates/kv/src/page.rs
task      debugging KV allocator

recent
✓ cargo test failed
✓ modified page.rs
✓ searched "pinned cuda memory"

ignored
✗ password manager
✗ private browser window
```

The context engine should never become an invisible black box.

---

# 21.3 Proactive suggestion engine

Marin should be able to surface useful suggestions without waiting for an explicit prompt.

The difficult problem is not:

> can Marin say something?

It is:

> should Marin say anything at all?

Silence should be the default.

## Proactivity pipeline

Use a cheap cascade.

```text
device event
    ↓
cheap deterministic rules
    ↓
tiny classifier / tiny model if needed
    ↓
potentially useful?
    ├── no → stop
    └── yes
          ↓
larger model only if needed
          ↓
suggestion
```

Most events should never reach the expensive model.

Possible model hierarchy:

```text
NO MODEL
discard obvious noise

0.8B / tiny model
classify relevance

2B / 4B
simple suggestion

9B
hard reasoning
```

This gives model routing a real product reason to exist.

## Suggestion policy

Marin should optimize:

```text
usefulness
────────────
interruption
```

Possible controls:

```text
minimum confidence threshold
maximum unsolicited suggestions per hour
cooldown after dismissal
higher threshold for repeated suggestion types
per-workspace preferences
```

Local feedback can modify relevance:

```text
accepted suggestion
        ↓
increase relevance weight

dismissed repeatedly
        ↓
reduce or suppress
```

No complex RL system is required initially.

## Example

User activity:

```text
cargo test fails
edit page.rs
cargo test fails differently
search docs
return to page.rs
```

Marin may surface:

```text
page.rs now keeps the borrowed page across the async boundary.
that looks related to the new compiler error.

want me to inspect that path?
```

Bad proactive behavior:

```text
"I noticed you're using Zed. Want productivity tips?"
```

Do not build that.

## Explainability

Every proactive suggestion should be inspectable.

```bash
marin why
```

Example:

```text
why this suggestion?

1. cargo test failed
2. page.rs changed immediately before the failure
3. a previous failure in this project involved the same allocator
```

Marin should be able to explain:

```text
what event triggered this?
what context was used?
what memory was used?
what model handled it?
did any data leave the device?
```

---

# 21.4 Always-on inference budget

An always-available assistant creates an inference problem of its own:

> how do we keep Marin useful all day without constantly burning GPU, RAM, battery, and power?

The always-on runtime should have explicit budgets.

Possible states:

```text
IDLE
no large model loaded
event collection only
near-zero accelerator activity

LIGHT AWARENESS
tiny model resident
cheap relevance / routing tasks

ACTIVE
medium model loaded
interactive assistant work

HEAVY
large model loaded
coding / difficult reasoning
```

Possible transition:

```text
idle event
    ↓
cheap rule
    ↓
interesting?
    ├── no → stay idle
    └── yes
          ↓
wake tiny / medium model
          ↓
hard task?
              ├── no → answer
              └── yes → wake large model
```

This makes the following inference work directly useful:

```text
model hibernation
fast cold start
weight paging
KV persistence
prefix caching
quantization
model hot swapping
background priority scheduling
CPU vs GPU placement
power-aware routing
```

The always-on assistant should become a real stress test for Marin's runtime rather than a separate product layer.

---

# 21.5 Assistant modes

Marin should expose one coherent assistant with different modes, not several unrelated agents.

## Assistant mode

General explicit requests.

Examples:

```text
find this file
explain this
summarize these notes
run this command
remember what we decided
organize these files
continue what I was doing
```

## Coding mode

A serious target is replacing a meaningful amount of Codex / Claude Code usage.

Capabilities eventually include:

```text
repository understanding
search
structured edits
diff generation
shell
git
tests
build loops
compiler feedback
long-running commands
persistent project state
project memory
resume after interruption
```

Example:

```bash
marin ~/dev/project
```

```text
> fix the failing tests
```

The same project state should survive across sessions.

## Proactive mode

Marin quietly observes allowed device events and only surfaces something when usefulness crosses the configured threshold.

The user should always be able to disable this mode completely.

---

# 22. Benchmark methodology

Marin needs three benchmark layers.

## 22.1 Microbenchmarks

Examples:

```text
matmul
attention
RoPE
norm
dequantization
KV copy
KV lookup
```

## 22.2 Inference benchmarks

Examples:

```text
short prefill
long prefill
short decode
long decode
batch 1
multiple sequences
```

## 22.3 Real workloads

Eventually:

```text
coding
repo understanding
chat
tool-heavy tasks
long-context work
multi-turn sessions
```

The real workload is what keeps the project honest.

---

# 23. Reference engines

Marin should compare itself against existing engines.

Possible references:

```text
llama.cpp
vLLM
SGLang
MLC
```

Use them to answer:

```text
how do they solve this?
what tradeoff did they choose?
how far behind are we?
where does Marin behave differently?
```

Do not copy architecture blindly.

Do not reinvent solved problems without learning from existing implementations.

---

# 24. Assistant and coding workload

Once the runtime is good enough, build the full Marin assistant surface.

The assistant is not the engine.

It is a client of the runtime.

Coding is the first demanding workload because it naturally stresses:

```text
context
memory
tool use
latency
prefix reuse
long sessions
process management
recovery
```

A serious long-term target is replacing a meaningful amount of Codex / Claude Code usage locally.

Architecture:

```text
marin assistant
        │
        ├── assistant mode
        ├── coding mode
        └── proactive mode
        │
        ▼
context + memory + tools
        │
        ▼
marin server
        │
        ▼
marin inference runtime
```

Agent capabilities:

```text
chat
filesystem
shell
git
structured edits
test loops
repo understanding
persistent conversations
memory
session resume
```

The agent becomes a stress test for:

```text
long context
prefix reuse
repeated sessions
tool outputs
background work
latency
memory pressure
```

---

# 25. Terminal UI / UX

The CLI remains important, but it belongs above the core runtime.

We eventually want two terminal surfaces:

```text
marin-run
    low-level inference/debug CLI

marin
    user-facing coding agent CLI
```

## marin-run UX

Focus on observability.

Example:

```text
model: tiny-transformer
backend: cuda
precision: fp16
ctx: 2k

> hello

hello...
```

Optional live metrics:

```text
TTFT       143 ms
prefill    2,310 tok/s
decode     81 tok/s
VRAM       2.1 GB
```

Useful commands:

```text
/metrics
/profile
/kv
/memory
/model
/backend
```

## coding agent UX

Later, make it feel similar to Codex / Claude Code in the terminal, but cleaner.

Principles:

```text
keyboard-first
low-noise
streaming
interruptible
diff-first
clear tool activity
fast session resume
scriptable
```

Tool calls should appear as compact events:

```text
› searched "scheduler" in crates/
  12 matches

› ran cargo test
  2 failed

› edited src/runtime.rs
  +14 -6
```

Do not dump every internal detail by default.

---

# 26. Memory layer for the coding agent

Memory remains useful, but it belongs to the application layer.

It should not be confused with inference memory such as KV cache.

Marin therefore has two different meanings of memory:

```text
INFERENCE MEMORY

weights
KV cache
runtime buffers
prefix cache
GPU/RAM/SSD state


AGENT MEMORY

conversation history
project knowledge
user preferences
previous decisions
workflow knowledge
```

Keep them architecturally separate.

## Agent memory hierarchy

```text
working memory
episodic memory
semantic memory
procedural memory
archive
```

## Initial storage

Prefer:

```text
SQLite
SQLite FTS5
metadata filtering
recency scoring
```

Embeddings can be added later.

Avoid requiring a heavyweight vector database.

## Memory properties

Agent memory must be:

```text
local
visible
editable
deletable
inspectable
scoped by project
```

The agent should expose:

```bash
marin memory list
marin memory search
marin memory inspect
marin memory forget
```

---

# 27. Phone + desktop

Phone support is part of the long-term inference-learning plan, but it should arrive in stages.

## Stage 1

```text
phone = client
laptop = runtime
```

This gives Marin an always-available interface without prematurely building mobile inference.

## Stage 2

```text
phone = client + small local runtime
laptop = stronger runtime
```

The phone becomes a fallback inference node.

## Stage 3

```text
multiple Marin nodes
        ↓
capability-aware routing
        ↓
portable sessions
```

The same session can move between devices while inference execution adapts to whatever hardware is available.

The runtime API should make clients independent from hardware, and the session format should remain model-independent wherever possible.

---

# 28. OpenAI-compatible server

Eventually:

```bash
marin serve
```

should expose an OpenAI-compatible API.

Then Marin can power:

```text
coding agent
terminal chat
web UI
phone app
editor plugins
scripts
existing AI clients
```

This gives us real workloads without coupling them to the engine.

---

# 29. Proposed repository structure

```text
marin/
│
├── crates/
│   ├── marin-core/
│   ├── marin-tensor/
│   ├── marin-loader/
│   ├── marin-model/
│   ├── marin-kv/
│   ├── marin-sampler/
│   ├── marin-quant/
│   ├── marin-scheduler/
│   ├── marin-speculate/
│   ├── marin-hw/
│   ├── marin-metrics/
│   ├── marin-server/
│   ├── marin-context/
│   ├── marin-policy/
│   └── marin-cli/
│
├── backends/
│   ├── cpu/
│   ├── cuda/
│   ├── vulkan/
│   └── metal/
│
├── apps/
│   ├── assistant/
│   ├── coding/
│   └── mobile/
│
├── benchmarks/
│   ├── kernels/
│   ├── inference/
│   └── workloads/
│
├── experiments/
│   ├── quantization/
│   ├── kv-cache/
│   ├── offload/
│   ├── scheduling/
│   ├── speculation/
│   ├── routing/
│   ├── always-on/
│   └── proactive/
│
└── docs/
    ├── architecture.md
    ├── model-support.md
    ├── benchmarks.md
    └── results/
```

---

# 30. Updated roadmap

```text
V0
correct transformer inference
        ↓
V1
instrument everything
        ↓
V2
execution optimization
        ↓
V3
quantization
        ↓
V4
paged / quantized KV
        ↓
V5
VRAM → RAM → SSD KV tiering
        ↓
V6
prefix reuse + persistent state
        ↓
V7
batching + scheduler
        ↓
V8
speculative decoding
        ↓
V9
hardware-adaptive runtime
        ↓
V10
multi-model routing
        ↓
V11
hibernation + recovery + migration
        ↓
V12
multi-device runtime
phone fallback
capability-aware routing
session handoff
        ↓
V13
situational context engine
structured device events
ephemeral context
privacy filters
        ↓
V14
always-on inference budget
model sleep/wake
cheap awareness
power-aware execution
        ↓
V15
proactive suggestion policy
relevance cascade
quiet-by-default behavior
explainable suggestions
        ↓
REAL APPLICATION

Marin assistant
assistant mode
coding mode
proactive mode
desktop + phone
OpenAI-compatible server
```

These versions are a north-star sequence, not a requirement to implement everything before Marin becomes usable.

The actual development loop remains:

```text
build one small thing
        ↓
understand it
        ↓
benchmark it
        ↓
use it in a real workload
        ↓
decide the next bottleneck
```

# 31. What we explicitly do not build early

Do not start with:

```text
multi-agent systems
24/7 screen recording
24/7 microphone capture
continuous OCR / vision over everything
fancy coding UI
giant memory layer
whole-life knowledge graph
browser automation before core context works
email/calendar integrations before they are useful
distributed inference
multi-node inference
fine-tuning
training models
huge MoE support
multi-GPU tensor parallelism
custom kernels for every operation
mobile inference before desktop runtime works
```

Those can come later if they actually help the core goal.

---

# 32. What success looks like

Early Marin:

```text
small model
FP16
simple KV
batch 1
single backend
basic generation
```

Later Marin:

```text
larger useful model
quantized weights
quantized KV
paged KV
GPU / RAM / SSD tiering
prefix reuse
continuous batching
smart scheduler
speculative decoding
automatic hardware planning
low-end hardware support
persistent inference state
```

And eventually:

```text
marin serve
```

powers the assistant we actually use every day.

That assistant should:

```text
act when explicitly asked
handle serious coding work
remember ongoing projects
understand allowed device context
surface genuinely useful suggestions
stay quiet most of the time
move between laptop and phone
keep private data local by default
```

The important difference is:

> the assistant exists because the runtime became efficient enough to stay useful all day on constrained hardware.

---

# 33. Final project statement

Marin is:

> a private, local-first intelligence layer built on a hardware-adaptive LLM inference runtime for constrained consumer hardware.

Its engineering core explores:

```text
execution
memory
quantization
KV cache
offloading
scheduling
speculation
hardware adaptation
continuity
heterogeneous inference
cross-device routing
state reconstruction
always-on inference
power-aware execution
```

Its product layer provides:

```text
assistant mode
coding mode
proactive mode
shared memory
situational context
desktop + phone continuity
```

Marin should not try to know everything.

It should try to know **just enough** to be useful.

It should not try to run the biggest model all the time.

It should use **the smallest amount of compute that can do the job well**.

It should not interrupt merely because it can.

It should speak only when:

```text
usefulness > interruption cost
```

And it should keep the user in control of:

```text
what Marin can observe
what Marin remembers
what model runs
where inference happens
what leaves the device
```

The final thesis is:

> Marin understands what you're doing, remembers enough to continue it, can act when asked, and knows when staying quiet is better, while aggressively minimizing compute, context, private data collection, and interruptions.

That is the project.
