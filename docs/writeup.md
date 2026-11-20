# Muse Spark serving: faster prefill MoE and TP-split pooler/unpooler projections

Final scored run: **1.457x** over stock SGLang (1 user 1.317x, 4 users 1.442x, 16 users 1.550x, 32 users 1.529x).
The patch serves Muse Spark exactly as stock does (TP8/EP8, NVFP4, DFlash, stock launch flags) and passes the
accuracy gate with headroom.

## Summary

The patch makes three changes to the prefill-heavy path:

- A W4A16 NVFP4 MoE kernel replaces Marlin, reading the same weight bytes. It is 2.2-2.9x faster at prefill sizes.
- The replicated pooler and unpooler projections are split across the 8 TP ranks: by token rows for prefill,
  by output columns for small decode and verify batches.
- A fix to the weight repack keeps peak memory at or below stock.

## Why prefill

The workload is agentic and multi-turn: about 92% of each prompt hits the prefix cache, but every turn still
prefills thousands of new tokens (tool outputs), and tok/s/GPU counts those input tokens. Prefill time dominates the
score at every load. A Marlin tile-size fix that sped up decode-size MoE calls moved the score only from 0.998x to
1.006x, so we spent effort where prefill time goes: the MoE at 1k-16k tokens per call, then the dense projections
that every TP rank computed in full.

## Changes

| Change | What it does | Where it applies | Measured effect |
| --- | --- | --- | --- |
| W4A16 MoE kernel | Routing, both expert GEMMs and the weighted sum in one Hopper kernel family, reading the checkpoint's NVFP4 weights after a lossless one-time repack; replaces Marlin | Every MoE call | 2.2-2.93x vs Marlin at prefill sizes; score 1.048x to 1.382x |
| 16,384-token MoE calls | The kernel takes up to 16,384 tokens per call instead of slicing | Large prefill batches | Part of 1.382x to 1.400x |
| Repack memory fix | Frees temporaries and the allocator cache around the repack | Model load | Peak memory at or below stock (was about +10 GB) |
| Row-split pooler/unpooler | Prefill batches of 1,024+ tokens are zero-padded to a multiple of TP; each rank projects 1/TP of the rows, then all-gather (pooler) or reduce-scatter, project, all-gather (unpooler) | Prefill | +1.4% at 16 users (at 8,192 tokens); +1.6% more with the 1,024 threshold |
| Column-split decode projections | Decode/verify batches of up to 64 tokens: each rank computes 1/TP of the output columns into a zeroed buffer; one all-reduce assembles them. Stock's fused gemm_ag path is kept where it applies (8 tokens or fewer) | 1 and 4 users | +4.0-4.5% at 1 and 4 users |

All gates are shape-based and identical on every rank (token count, forward mode, TP size, GPU capability), so every
rank takes the same branch and collectives stay matched. Each lever can be turned off with an environment variable:
`SGLANG_MUSE_W4A16_MOE_LAYERS`, `SGLANG_MUSE_SHARD_POOLER_MIN_TOKENS`, `SGLANG_MUSE_SHARD_DECODE_MAX_TOKENS`,
`SGLANG_MUSE_SHARD_UNPOOLER`.

## Correctness

Final run, over 176 reference turns:

| Gate | Result | Limit |
| --- | --- | --- |
| Prefill KL | 0.049 | 0.11 |
| Decode KL | 0.014 | 0.035 |
| Matching prefix | 0.14 | 0.05 minimum |
| Later-turn agreement (rescored by stock) | 0.97 | 0.92 minimum |
| GPU memory vs stock | +0.0 GB | +2 GB |
| TTFT p90 vs stock (4- and 16-user checks) | 0.62x first turn, 0.70-0.76x later | 2x |

- **Prefill row split is bitwise stock.** On one H200 with the judge image, every rank's padded row slice of the
  pooler and unpooler GEMM matched stock's full GEMM bit for bit at 1,024, 1,025, 3,012, 4,097, 8,191 and 16,384
  tokens. The all-gather only copies rows.
- **Decode column split differs only by rounding.** Its 1/TP-width GEMMs select a different cuBLAS kernel, so outputs
  differ by up to two bf16 rounding steps. Decode KL did not move (0.0094-0.014 across runs).
- **MoE kernel.** FP32 relative error 0.0029 against a reference, versus 0.0056 for Marlin; CUDA-graph replay exact.
- **CPU tests.** 115 tests cover padding, multi-rank emulations of the row and column splits, gate conditions and the MoE runtime.

## Results

The final run improved every load over the previous best. Stock ran within about 2% of its speed in the previous best
run at every load, and draft acceptance was equal or lower, so the gain is not baseline or drafting luck.

| Load | Previous best (1.400x run) | Final (1.457x run) | Throughput, tok/s/GPU | Gain |
| --- | --- | --- | --- | --- |
| 1 user | 1.237x | 1.317x | 7,344 to 7,671 | +4.5% |
| 4 users | 1.380x | 1.442x | 12,673 to 13,181 | +4.0% |
| 16 users | 1.508x | 1.550x | 17,925 to 18,519 | +3.3% |
| 32 users | 1.493x | 1.529x | 18,893 to 19,417 | +2.8% |

Scored-run history: Nov 12 0.998x · Nov 13 1.006x (Marlin tile fix) · Nov 14 1.036x (pooler split) · Nov 15 1.048x
(unpooler split) · Nov 16 1.041x (8k chunks, dropped) · Nov 17 1.382x (MoE kernel) · Nov 18 1.400x (16k MoE calls,
cleanup) · Nov 19 1.457x (TP row/column splits).

## What did not work

- 8,192- and 32,768-token prefill chunks (8,192 scored 1.041x against 1.048x)
- Copy-tier drafting (-5% at 4 users) and 16 draft tokens
- NCCL NVLS for prefill collectives, more fast-MoE weight banks, the static memory fraction, KV-cache levers
- MoE runtime knob tuning (within 0.2%)

Measurement: stock's speed varies 3-8% between nodes and draft acceptance moves throughput 2-5%, so we compared
patches by absolute tok/s/GPU at matched acceptance, not by single ratios.

## Reproduction

| Item | Value |
| --- | --- |
| Base | SGLang 297e54fd (release v0.5.23), judge image sha256:f832b5c3 |
| Branch / commit | `levers-final` at `d2e7a0db` |
| Patch | `serving/patch-levers-final.diff`, 121,093 bytes, 10 files under `python/sglang` |
| Scored run | job `6ac30c2d11a749d4` (1.457x) |
| Checks | `a9fbb0fae85a4b3a` (4 users), `d67b25bd60e14ca1` (16 users); control `ed756d587e7a4a85` |
| Launch flags | stock's, unchanged |
