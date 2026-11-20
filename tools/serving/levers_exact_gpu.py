"""1xH200 bitwise check: do TP8 row/column slices of the Muse pooler/unpooler GEMMs equal the
corresponding rows/columns of stock's full (replicated) GEMM?

The levers' collectives only copy rows (all_gather) or add zeros (column all-reduce), so slice
equality implies the sharded projection output is bitwise stock. No collectives run here.

    tools/serving/modal_env.sh run tools/serving/levers_exact_gpu.py
"""

from __future__ import annotations

import json
from pathlib import Path

PREFILL_TOKENS = (1024, 1025, 3012, 4097, 8191, 16384)
DECODE_TOKENS = (8, 32, 64)
HIDDEN, POOLED, WORLD = 8192, 4096, 8

try:
    import modal
except ModuleNotFoundError:
    modal = None

if modal is not None:
    _HERE = Path(__file__).resolve()
    ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/")
    IMAGE = "lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
    app = modal.App("levers-exact")
    image = modal.Image.from_registry(IMAGE)
    if modal.is_local():
        image = image.add_local_file(str(ROOT / "patch/levers-final.diff"), "/proj/patch.diff")

    @app.function(image=image, gpu="H200", cpu=4, memory=32768, timeout=240, retries=0,
                  scaledown_window=2)
    def run_exact() -> dict:
        import subprocess
        import sys

        import torch
        import torch.nn.functional as F

        src = "/sgl-workspace/sglang"
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
        assert head.startswith("297e54fd"), head
        subprocess.run(["git", "apply", "/proj/patch.diff"], cwd=src, check=True)
        sys.path.insert(0, src + "/python")
        from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch as gemm

        def rand(shape, seed):
            g = torch.Generator(device="cuda").manual_seed(seed)
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)

        def compare(parts):
            worst, equal = 0.0, True
            for local, ref in parts:
                equal &= bool(torch.equal(local, ref))
                worst = max(worst, (local.float() - ref.float()).abs().max().item())
            return {"bitwise": equal, "max_abs": worst}

        def row_slices(x, w):
            full = gemm(x, w, None)
            padded = F.pad(x, (0, 0, 0, (-x.shape[0]) % WORLD))
            rows = padded.shape[0] // WORLD
            parts = []
            for r in range(WORLD):
                local = gemm(padded[r * rows:(r + 1) * rows], w, None)
                ref = full[r * rows:(r + 1) * rows]
                parts.append((local[:ref.shape[0]], ref))
            return compare(parts)

        def column_slices(x, w):
            full = gemm(x, w, None)
            cols = w.shape[0] // WORLD
            return compare([(gemm(x, w[r * cols:(r + 1) * cols], None), full[:, r * cols:(r + 1) * cols])
                            for r in range(WORLD)])

        w_pool, w_unpool = rand((POOLED, HIDDEN), 1), rand((HIDDEN, POOLED), 2)
        result = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                  "prefill_rows": [], "decode_columns": []}
        for t in PREFILL_TOKENS:
            result["prefill_rows"].append({
                "T": t, "pool": row_slices(rand((t, HIDDEN), 100 + t), w_pool),
                "unpool": row_slices(rand((t, POOLED), 200 + t), w_unpool)})
        for t in DECODE_TOKENS:
            result["decode_columns"].append({
                "T": t, "pool": column_slices(rand((t, HIDDEN), 300 + t), w_pool),
                "unpool": column_slices(rand((t, POOLED), 400 + t), w_unpool)})
        print(json.dumps(result), flush=True)
        return result

    @app.local_entrypoint()
    def main():
        result = run_exact.remote()
        path = ROOT / "notes/serving/levers-exact.json"
        path.write_text(json.dumps(result, indent=2))
        print("saved", path)
