# gpu-planner

[![ci](https://github.com/bugyal/gpu-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/bugyal/gpu-planner/actions/workflows/ci.yml)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Answers two questions people keep redoing by hand in forum threads:

1. **Can I run model X on my hardware?** — per-quant feasibility, VRAM breakdown,
   a layer-offload plan, and a tokens/sec estimate.
2. **When does renting beat buying?** — a break-even sheet against cloud hourly
   rates, in totals rather than vibes.

No runtime dependencies. Python 3.11+.

## Install

```sh
/opt/homebrew/bin/python3.13 -m venv .venv   # or any python3.11+
.venv/bin/pip install -e ".[dev]"            # [dev] adds pytest
.venv/bin/gpu-planner --list
```

`python3` on macOS is often the 3.9 system build, which has no `tomllib` and a
pip too old for editable installs — name a 3.11+ interpreter explicitly.

## Use

```sh
# Ask about one model on a card you name
gpu-planner --gpu rtx-4090-24g --ram 64 --model llama-3.3-70b --quant q4_k_m

# Compare quants
gpu-planner --gpu rtx-4090-24g --ram 64 --model llama-3.3-70b \
            --quant q4_k_m --quant q3_k_m --quant q2_k

# Use your own machine's detected hardware
gpu-planner --model llama-3.1-8b

# Declare a rig
gpu-planner --inventory examples/my-rig.toml --model mixtral-8x7b --context 32768

# Include the rent-vs-buy sheet
gpu-planner --gpu rtx-4090-24g --ram 64 --model llama-3.1-8b --rent --hours 300

# Machine-readable, and the catalogue
gpu-planner --gpu h200-141g --ram 256 --model deepseek-v3 --json
gpu-planner --list
```

Flags: `--model` / `--quant` repeat, `--context` (default 8192) sets the window,
`--gpu/--vram/--ram/--bandwidth/--cpu` override detection, `--inventory` reads a
small TOML file, `--rent` adds the break-even sheet with `--price`,
`--cloud-rate`, `--hours`, `--amortise-months`, `--watts`.

Exit codes: `0` ok, `4` bad input.

## Example — a 70B that does not fit a 24GB card

```
$ gpu-planner --gpu rtx-3090-24g --ram 128 --model llama-3.3-70b \
              --quant q4_k_m --quant q3_k_m --context 8192
inventory:
  os:        Darwin 24.3.0
  cpu:       4 cores
  ram:       128GB
  gpu:       rtx-3090-24g (24GB, discrete)
  bandwidth: 936 GB/s

llama-3.3-70b  (70.6B total, 80 layers)

  quant       weights       kv    over    total verdict              tok/s
  ------------------------------------------------------------------------
  q4_k_m       42.75G    2.68G   1.50G   46.93G needs CPU overspill     2.4
  q3_k_m       34.47G    2.68G   1.33G   38.49G needs CPU overspill     3.6

  nothing fits fully on GPU at 8192 ctx — see the overspill plan

  overspill plan (80 layers):
    offload 41/80 layers to CPU (49% on GPU, 39 layers); 23.28GB moved off the GPU
    max context fully on GPU: 0 tokens
    max context using GPU+RAM: 328,839 tokens

  throughput (q4_k_m):
    2.4 tok/s — bottleneck: CPU/RAM stage (layers offloaded)
    41/80 layers on CPU; GPU-only would be 12.0 tok/s
```

That is the arithmetic the threads keep redoing: 42.75GB of weights at Q4_K_M
plus 2.68GB of KV cache at 8k plus ~1.5GB of runtime overhead is 46.9GB against
24GB of VRAM, so ~41 of 80 layers have to move to CPU, and throughput collapses
from ~12 tok/s to ~2.4.

## Example — same model, bigger card, plus rent vs buy

```
$ gpu-planner --gpu h100-80g --ram 256 --model llama-3.3-70b --quant q4_k_m --rent --hours 300
```

The feasibility table reports `fits fully`, and the rent-vs-buy section compares
the amortised owning cost against the cloud rate, with a break-even in hours and
months and a cost per million tokens on each side.

## Example — a MoE model, where total and active parameters differ

```
$ gpu-planner --inventory examples/my-rig.toml --model mixtral-8x7b --quant q4_k_m --context 32768
mixtral-8x7b  (46.7B total, 12.9B active, 32 layers)
  note: MoE: 46.7B total, ~12.9B active per token
```

The weights are sized from the **total** parameters — all experts have to be
resident — but throughput is estimated from the **active** parameters, because
only the active experts are read per token. Conflating the two is the usual way
these estimates go wrong.

## How the numbers are computed

**Weights.** `params x bytes_per_param(quant)`, where bytes-per-parameter comes
from the published llama.cpp k-quant table (Q4_K_M = 4.8438 bits/weight, Q8_0 =
8.5, fp16 = 16).

**KV cache.** `2 (K and V) x layers x kv_heads x head_dim x dtype_bytes` per
token, times the context window. At fp16 a 70B (80 layers, 8 kv heads, 128 dim)
is 2.68GB per 8k tokens — which is why long context is what actually OOMs people,
not the weights.

**Runtime overhead.** A 0.6GB floor plus 2% of the weight bytes, covering the
CUDA context, cuBLAS workspaces, and fragmentation.

**Throughput.** Decode is memory-bandwidth bound, so
`tokens/s = bandwidth x efficiency / active_weight_bytes` with an efficiency of
0.55 for discrete GPUs and 0.70 for unified memory. When layers are offloaded the
per-token latency becomes the GPU stage plus the CPU stage (at ~100GB/s of
realistic RAM bandwidth) plus a per-layer transfer penalty.

**Break-even.** Totals, not rates. Over `H` hours, owning costs
`price + electricity x H` and renting costs `cloud_rate x H`; they are equal at
`H = price / (cloud_rate - electricity_rate)`. If electricity alone exceeds the
cloud rate, owning never pays back and the tool says so rather than printing an
absurd break-even.

Units: memory is reported in decimal GB, matching how GPUs are sold. The two
exceptions are visible in the code rather than silently mixed.

## Why a local CLI and not a web calculator

Free web calculators for this already exist (benchlm.ai, aimultiple,
erans/selfhostllm, localaimaster), and for a quick one-off lookup they are fine.
gpu-planner differs on four points:

- **Local CLI, no browser tab.** It runs in a shell, exits non-zero on bad
  input, and prints JSON with `--json` — so it can sit inside an agent pipeline
  or a one-liner (`gpu-planner ... | jq`) the way a web form cannot.
- **Hardware auto-detect.** It reads the actual machine — GPU model and VRAM,
  RAM, memory bandwidth — instead of asking you to fill a form from memory.
  Any detected value can be overridden with a flag or a small inventory file.
- **A layer-offload overspill plan.** When weights do not fit, it computes the
  per-layer split (how many of the 80 layers move to CPU, how much VRAM that
  frees, the max context that still fits) — arithmetic the web forms do not
  attempt. They stop at "does not fit".
- **The math is shown with its assumptions.** The throughput model and the
  rent-vs-buy break-even are documented formulas with stated efficiency
  factors and limitations, not a black box: you can check why a number came
  out the way it did.

## Honest limitations

- **Estimates, not measurements.** The throughput model gets the order of
  magnitude and the effect of quantisation and offload right. Kernel quality,
  batch size, and prompt-processing speed are not modelled — treat the numbers as
  a sanity check, not a benchmark.
- **Single GPU.** Multi-GPU and tensor-parallel setups are not modelled.
- **Overhead is a heuristic.** Real runtime overhead varies by backend and
  driver version.
- **The catalogue is small and hand-entered.** Unknown models can be given as a
  bare size (`--model 13b`), and the geometry is then estimated — the output
  says so. Unknown GPUs need `--vram` and `--bandwidth`.
- **Cloud rates are approximate public list prices.** Consumer cards have no
  hourly rate, so when `--rent` is used on one the tool substitutes the nearest
  datacentre class and labels the substitution instead of guessing silently.
- **AMD detection is thin.** `rocm-smi` is probed but the card name and
  bandwidth are not resolved; pass them explicitly.

## Tests

64 tests. The VRAM maths is pinned by computing the expected values from first
principles in the test (weights, KV per token, overhead, throughput, break-even
crossover) so a change that alters the meaning of a figure fails there, not in
production.

```sh
.venv/bin/python -m pytest
```

## License

Apache-2.0 — see `LICENSE`.
