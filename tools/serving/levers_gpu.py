"""Single-H200 projection and verify-size GEMM timing for Muse Spark; no model download."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


WORLD_SIZE = 8
HIDDEN_SIZE = 8192
POOLED_SIZE = 4096
LAYERS = 62
TOKENS = (1024, 3012, 8192)
VERIFY_TOKENS = (8, 32, 64)

# Eager NCCL measurements supplied for the 8xH200 deployment, in microseconds.
COLLECTIVE_ANCHORS = {
    "ar_4096": ((1024, 74.8), (4096, 210.6), (16384, 702.7)),
    "ag_4096": ((1024, 51.5), (4096, 123.5), (16384, 387.8)),
    "ag_8192": ((1024, 66.9), (4096, 204.9), (16384, 733.7)),
}

# MuseSparkTextConfig defaults plus MuseSpark1xAttention's TP-local dimensions:
# Q=128*64, KV=16*64, and the output gate repeats Q. At TP8 this is
# (Q + K + V + gate) / 8 = 2304 outputs; o_proj consumes Q / 8 = 1024.
VERIFY_WEIGHT_SHAPES = {
    "qkv_gate": (2304, HIDDEN_SIZE),
    "o_proj": (HIDDEN_SIZE, 1024),
    "router_hi": (256, HIDDEN_SIZE),
    "router_lo": (256, HIDDEN_SIZE),
    "pooler": (POOLED_SIZE, HIDDEN_SIZE),
    "unpooler": (HIDDEN_SIZE, POOLED_SIZE),
}


def interpolate_collectives(tokens: int) -> dict[str, float]:
    """Linearly interpolate supplied collectives, clamping T below 1024."""
    if tokens < 0:
        raise ValueError(f"tokens must be non-negative, got {tokens}")

    query = max(tokens, 1024)
    result: dict[str, float] = {}
    for name, anchors in COLLECTIVE_ANCHORS.items():
        if query > anchors[-1][0]:
            raise ValueError(
                f"no {name} collective anchor above T={anchors[-1][0]}"
            )
        for (left_t, left_us), (right_t, right_us) in zip(anchors, anchors[1:]):
            if left_t <= query <= right_t:
                fraction = (query - left_t) / (right_t - left_t)
                result[name] = left_us + fraction * (right_us - left_us)
                break
    result["rs_4096"] = 0.5 * result["ar_4096"]
    return result


def _timing_us(value: Any) -> float:
    if isinstance(value, Mapping):
        value = value["us"]
    timing = float(value)
    if timing < 0:
        raise ValueError(f"timing must be non-negative, got {timing}")
    return timing


def compute_savings(
    timing_by_tokens: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine measured GEMMs with the historical collective interpolation."""
    rows: list[dict[str, Any]] = []
    saving_names = (
        "col_pool",
        "row_pool",
        "shard_unpool",
        "col_pool_plus_shard_unpool",
        "row_pool_plus_shard_unpool",
    )

    for tokens in sorted(timing_by_tokens):
        measured = timing_by_tokens[tokens]
        collective = interpolate_collectives(tokens)
        stock_pool = _timing_us(measured["pool_full"])
        col_pool = (
            _timing_us(measured["pool_cols"])
            + _timing_us(measured["zero_fill"])
            + collective["ar_4096"]
        )
        row_pool = _timing_us(measured["pool_rows"]) + collective["ag_4096"]
        stock_unpool = collective["ar_4096"] + _timing_us(
            measured["unpool_full"]
        )
        shard_unpool = (
            collective["rs_4096"]
            + _timing_us(measured["unpool_rows"])
            + collective["ag_8192"]
        )
        saving_us = {
            "col_pool": stock_pool - col_pool,
            "row_pool": stock_pool - row_pool,
            "shard_unpool": stock_unpool - shard_unpool,
        }
        saving_us["col_pool_plus_shard_unpool"] = (
            saving_us["col_pool"] + saving_us["shard_unpool"]
        )
        saving_us["row_pool_plus_shard_unpool"] = (
            saving_us["row_pool"] + saving_us["shard_unpool"]
        )
        rows.append(
            {
                "T": tokens,
                "collective_us": collective,
                "cost_us_per_layer": {
                    "stock_pool": stock_pool,
                    "col_pool": col_pool,
                    "row_pool": row_pool,
                    "stock_unpool": stock_unpool,
                    "shard_unpool": shard_unpool,
                },
                "savings": {
                    name: {
                        "us_per_layer": saving_us[name],
                        "ms_x62": saving_us[name] * LAYERS / 1000.0,
                    }
                    for name in saving_names
                },
            }
        )

    crossovers = {
        name: next(
            (
                row["T"]
                for row in rows
                if row["savings"][name]["us_per_layer"] > 0
            ),
            None,
        )
        for name in saving_names
    }
    return {"rows": rows, "crossovers": crossovers}


def postprocess_gpu_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Create the JSON document written by the local Modal entrypoint."""
    timing_by_tokens = {int(row["T"]): row for row in raw["projection_timings"]}
    if set(timing_by_tokens) != set(TOKENS):
        raise ValueError(
            f"expected projection timings for {TOKENS}, got {tuple(timing_by_tokens)}"
        )
    return {
        "schema_version": 1,
        "benchmark": dict(raw),
        "analysis": compute_savings(timing_by_tokens),
        "collective_model": {
            "description": (
                "Linear interpolation of supplied eager-NCCL 8xH200 timings; "
                "T below 1024 uses the 1024 value; RS4096 is 0.5*AR4096."
            ),
            "anchors_us": COLLECTIVE_ANCHORS,
        },
    }


def render_markdown(document: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to the JSON result."""
    benchmark = document["benchmark"]
    analysis = document["analysis"]
    lines = [
        "# Projection timing",
        "",
        f"GPU: {benchmark['gpu']}  ",
        f"SGLang HEAD: `{benchmark['head']}`  ",
        "Times are median microseconds. Collective entries are interpolated "
        "from the supplied eager-NCCL 8xH200 measurements, not measured here.",
        "",
        "## Projection alternatives",
        "",
        "| T | stock pool | col pool | row pool | stock unpool | shard unpool |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["rows"]:
        cost = row["cost_us_per_layer"]
        lines.append(
            f"| {row['T']} | {cost['stock_pool']:.2f} | {cost['col_pool']:.2f} "
            f"| {cost['row_pool']:.2f} | {cost['stock_unpool']:.2f} "
            f"| {cost['shard_unpool']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Savings",
            "",
            "Each cell is `us/layer; ms x62`. Positive values favor the alternative.",
            "",
            "| T | col pool | row pool | shard unpool | col+unpool | row+unpool |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    saving_order = (
        "col_pool",
        "row_pool",
        "shard_unpool",
        "col_pool_plus_shard_unpool",
        "row_pool_plus_shard_unpool",
    )
    for row in analysis["rows"]:
        cells = []
        for name in saving_order:
            saving = row["savings"][name]
            cells.append(f"{saving['us_per_layer']:+.2f}; {saving['ms_x62']:+.3f}")
        lines.append(f"| {row['T']} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Crossovers", ""])
    for name in saving_order:
        value = analysis["crossovers"][name]
        lines.append(f"- `{name}`: {value if value is not None else 'none in sweep'}")

    lines.extend(
        [
            "",
            "## Verify-size dense GEMMs",
            "",
            "| T | projection | input | weight | us | mode |",
            "|---:|---|---|---|---:|---|",
        ]
    )
    for row in benchmark["verify_dense"]:
        timing = row["timing"]
        lines.append(
            f"| {row['T']} | {row['name']} | `{row['input_shape']}` | "
            f"`{row['weight_shape']}` | {timing['us']:.2f} | {timing['mode']} |"
        )
    return "\n".join(lines) + "\n"


try:
    import modal
except ModuleNotFoundError:
    modal = None


if modal is not None:
    _HERE = Path(__file__).resolve()
    ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/")
    IMAGE = (
        "lmsysorg/sglang@sha256:"
        "f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f"
    )
    SOURCE = "/sgl-workspace/sglang"
    TIMEOUT_SECONDS = 420
    USD_CAP = 0.60
    USD_PER_SECOND = 0.001261 + 4 * 0.0000131 + 32 * 0.00000222
    SPEND_LOG = ROOT / "notes/h200/spend-2026-11-09.jsonl"

    app = modal.App("levers-gpu")
    image = modal.Image.from_registry(IMAGE)
    if modal.is_local():
        image = image.add_local_file(
            str(ROOT / "patch/levers-final.diff"), "/proj/patch.diff"
        )
    cache = modal.Volume.from_name("okc-ext-cache", create_if_missing=True)

    @app.function(
        image=image,
        gpu="H200",
        cpu=4,
        memory=32768,
        timeout=TIMEOUT_SECONDS,
        retries=0,
        volumes={"/cache": cache},
        scaledown_window=2,
    )
    def run_gpu(label: str) -> dict[str, Any]:
        import gc
        import statistics
        import subprocess
        import sys
        import time
        import traceback

        import torch
        import torch.nn.functional as F

        if not label or not all(c.isalnum() or c in "-_" for c in label):
            raise ValueError("label must contain only letters, digits, '-' or '_'")

        started = time.monotonic()
        result: dict[str, Any] = {
            "label": label,
            "projection_timings": [],
            "verify_dense": [],
            "verify_weight_shapes": VERIFY_WEIGHT_SHAPES,
        }
        result_path = Path(f"/cache/results/levers-gpu-{label}.json")

        def save(phase: str) -> None:
            result["phase"] = phase
            result["gpu_seconds"] = time.monotonic() - started
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            cache.commit()
            print(
                "PHASE",
                phase,
                round(result["gpu_seconds"], 2),
                flush=True,
            )

        def measure(fn: Any) -> dict[str, Any]:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()

            graph = None
            graph_output = None
            capture_error = None
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = fn()

                def launch() -> None:
                    graph.replay()

                mode = "graph"
            except RuntimeError as error:
                graph = None
                graph_output = None
                capture_error = f"{type(error).__name__}: {error}"
                torch.cuda.synchronize()

                def launch() -> None:
                    fn()

                mode = "eager-fallback"

            launch()
            torch.cuda.synchronize()
            samples = []
            for _ in range(10):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):
                    launch()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000.0 / 20.0)

            output = graph_output if graph_output is not None else fn()
            if not torch.isfinite(output).all().item():
                raise ValueError("nonfinite benchmark output")
            timing = {
                "us": statistics.median(samples),
                "mode": mode,
                "samples_us": samples,
            }
            if capture_error is not None:
                timing["capture_error"] = capture_error
            return timing

        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
            ).strip()
            assert head.startswith("297e54fd"), head
            subprocess.run(
                ["git", "apply", "--check", "/proj/patch.diff"],
                cwd=SOURCE,
                check=True,
            )
            subprocess.run(
                ["git", "apply", "/proj/patch.diff"], cwd=SOURCE, check=True
            )
            sys.path.insert(0, SOURCE + "/python")
            from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

            generator = torch.Generator(device="cuda").manual_seed(56)
            bf16 = torch.bfloat16
            pool_weight = torch.randn(
                POOLED_SIZE,
                HIDDEN_SIZE,
                device="cuda",
                dtype=bf16,
                generator=generator,
            )
            unpool_weight = torch.randn(
                HIDDEN_SIZE,
                POOLED_SIZE,
                device="cuda",
                dtype=bf16,
                generator=generator,
            )
            result.update(
                head=head,
                gpu=torch.cuda.get_device_name(),
                torch=str(torch.__version__),
            )
            save("patched")

            for tokens in TOKENS:
                if time.monotonic() - started > TIMEOUT_SECONDS - 30:
                    raise TimeoutError("stopping before the Modal function timeout")
                x = torch.randn(
                    tokens,
                    HIDDEN_SIZE,
                    device="cuda",
                    dtype=bf16,
                    generator=generator,
                )
                y = torch.randn(
                    tokens,
                    POOLED_SIZE,
                    device="cuda",
                    dtype=bf16,
                    generator=generator,
                )
                row_count = tokens // WORLD_SIZE
                row = {
                    "T": tokens,
                    "rows": row_count,
                    "pool_full": measure(
                        lambda: bf16_gemm_dispatch(x, pool_weight, None)
                    ),
                    "pool_rows": measure(
                        lambda: bf16_gemm_dispatch(x[:row_count], pool_weight, None)
                    ),
                    "pool_cols": measure(
                        lambda: F.linear(x, pool_weight[: POOLED_SIZE // WORLD_SIZE])
                    ),
                    "zero_fill": measure(
                        lambda: x.new_zeros((tokens, POOLED_SIZE))
                    ),
                    "unpool_full": measure(
                        lambda: bf16_gemm_dispatch(y, unpool_weight, None)
                    ),
                    "unpool_rows": measure(
                        lambda: bf16_gemm_dispatch(
                            y[:row_count], unpool_weight, None
                        )
                    ),
                }
                result["projection_timings"].append(row)
                print(json.dumps(row), flush=True)
                save(f"projection-T{tokens}")
                del x, y, row
                gc.collect()
                torch.cuda.empty_cache()

            qkv_gate_weight = torch.randn(
                *VERIFY_WEIGHT_SHAPES["qkv_gate"],
                device="cuda",
                dtype=bf16,
                generator=generator,
            )
            o_proj_weight = torch.randn(
                *VERIFY_WEIGHT_SHAPES["o_proj"],
                device="cuda",
                dtype=bf16,
                generator=generator,
            )
            router_fp32 = torch.randn(
                256,
                HIDDEN_SIZE,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
            router_hi = router_fp32.to(bf16)
            router_lo = (router_fp32 - router_hi.float()).to(bf16)
            verify_weights = {
                "qkv_gate": qkv_gate_weight,
                "o_proj": o_proj_weight,
                "router_hi": router_hi,
                "router_lo": router_lo,
                "pooler": pool_weight,
                "unpooler": unpool_weight,
            }
            import importlib
            importlib.import_module("sglang.srt.models.muse_spark")
            result["import_smoke"] = "sglang.srt.models.muse_spark imported"
            verify_weights["pooler_col_shard"] = pool_weight[: POOLED_SIZE // WORLD_SIZE].contiguous()
            verify_weights["unpooler_col_shard"] = unpool_weight[: HIDDEN_SIZE // WORLD_SIZE].contiguous()
            for tokens in VERIFY_TOKENS:
                for name, weight in verify_weights.items():
                    input_tensor = torch.randn(
                        tokens,
                        weight.shape[1],
                        device="cuda",
                        dtype=bf16,
                        generator=generator,
                    )
                    record = {
                        "T": tokens,
                        "name": name,
                        "input_shape": list(input_tensor.shape),
                        "weight_shape": list(weight.shape),
                        "timing": measure(
                            lambda input_tensor=input_tensor, weight=weight: (
                                bf16_gemm_dispatch(input_tensor, weight, None)
                            )
                        ),
                    }
                    result["verify_dense"].append(record)
                    print(json.dumps(record), flush=True)
            save("complete")
            return result
        except BaseException:
            result["error"] = traceback.format_exc()
            save("failed")
            raise

    @app.local_entrypoint()
    def main(label: str = "r1", run_timeout: int = TIMEOUT_SECONDS) -> None:
        import time

        if not 0 < run_timeout <= TIMEOUT_SECONDS:
            raise ValueError(f"run_timeout must be in [1, {TIMEOUT_SECONDS}]")
        worst_case_usd = run_timeout * USD_PER_SECOND
        if worst_case_usd > USD_CAP:
            raise RuntimeError(
                f"worst-case cost ${worst_case_usd:.3f} exceeds ${USD_CAP:.2f} cap"
            )

        started = time.monotonic()
        try:
            raw = run_gpu.with_options(timeout=run_timeout).remote(label)
            document = postprocess_gpu_result(raw)
            output_dir = ROOT / "notes/serving"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "levers-gpu.json").write_text(
                json.dumps(document, indent=2), encoding="utf-8"
            )
            (output_dir / "levers-gpu.md").write_text(
                render_markdown(document), encoding="utf-8"
            )
        finally:
            elapsed = time.monotonic() - started
            entry = {
                "label": f"levers-gpu-{label}",
                "seconds": elapsed,
                "usd": min(run_timeout, elapsed) * USD_PER_SECOND,
                "note": (
                    f"conservative bound, function timeout {run_timeout}s; "
                    "client wall includes allocation"
                ),
            }
            SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
            with SPEND_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

