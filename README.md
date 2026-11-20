# Muse Spark serving: 1.457x over stock SGLang

Our serving-track entry for the [Open Kernel Challenge 2026](https://openkernelchallenge.dev): a patch to SGLang
that serves Meta's Muse Spark Open (816B MoE, NVFP4 experts) on 8x H200 **1.457x** faster than stock SGLang,
measured by the challenge judge at stock's per-user speed, losslessly, with stock's launch flags.

| Users | 1 | 4 | 16 | 32 | Geometric mean |
| --- | --- | --- | --- | --- | --- |
| Speedup vs stock | 1.317x | 1.442x | 1.550x | 1.529x | **1.457x** |

## What the patch does

The workload is agentic and multi-turn, so prefill dominates the score. The patch:

- **Replaces Marlin with a W4A16 NVFP4 MoE kernel for Hopper** (routing, both expert GEMMs and the weighted sum,
  reading the checkpoint's weights after a lossless repack): 2.2-2.9x faster than Marlin at prefill sizes, same
  weight bytes, peak memory at or below stock.
- **Splits the replicated pooler and unpooler projections across TP ranks:** by token rows for prefill
  (bitwise identical to stock), by output columns for small decode and verify batches.

The full write-up, with the correctness evidence and what did not work, is in [docs/writeup.md](docs/writeup.md).

## Layout

| Path | Contents |
| --- | --- |
| `patch/levers-final.diff` | The scored patch, against SGLang `297e54fd` (release v0.5.23) |
| `kernel/w4a16_moe/` | The MoE kernel sources from the patch, for reading (CUDA, bindings, JIT build, runtime, weight packing) |
| `docs/` | Write-up and measured results (bitwise exactness test, projection timings) |
| `tools/serving/` | CPU tests for the patch, the patch-rule validator, and single-H200 Modal harnesses |

## Reproduce

```bash
git clone https://github.com/sgl-project/sglang serving/sglang
git -C serving/sglang checkout 297e54fd105e974cdf401e5b4d20253a22c16f90
git -C serving/sglang apply ../../patch/levers-final.diff

# CPU tests (115): need torch (CPU build) and pytest
python -m pytest -q tools/serving/test_*.py
```

The GPU harnesses (`tools/serving/levers_*_gpu.py`) run on one H200 through [Modal](https://modal.com) against the
judge's SGLang image.

## License

Apache License 2.0. The MoE kernel derives from Apache-2.0 kernel-challenge work by vjeux and alygg77; see
[NOTICE](NOTICE).
