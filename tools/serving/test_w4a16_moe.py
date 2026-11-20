"""CPU contract tests; real norm class definitions, no SGLang dependency imports."""

import ast
import enum
import importlib
import math
import os
from pathlib import Path
import sys
import types
from typing import Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SGL = Path(os.environ.get("SGLANG_TEST_CHECKOUT", ROOT / "serving/sglang")) / "python"
for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.muse_spark"):
    mod = types.ModuleType(name)
    mod.__path__ = [str(SGL.joinpath(*name.split(".")))]
    sys.modules[name] = mod
name = "sglang.srt.layers.muse_spark.muse_spark_layers"
norms = types.ModuleType(name)
norms.__dict__.update(
    torch=torch,
    nn=torch.nn,
    F=torch.nn.functional,
    enum=enum,
    math=math,
    Optional=Optional,
    _is_cpu=False,
)
source = ast.parse((SGL / "sglang/srt/layers/muse_spark/muse_spark_layers.py").read_text())
selected = [
    n
    for n in source.body
    if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    and n.name
    in (
        "GainCenterType",
        "calculate_gain_center",
        "MuseSpark1xRMSNorm",
        "MuseSpark1xRMSNormWithInputScale",
    )
]
exec(
    compile(ast.Module(selected, type_ignores=[]), "<real norm definitions>", "exec"),
    norms.__dict__,
)
sys.modules[name] = norms
runtime = importlib.import_module("sglang.srt.layers.muse_spark.w4a16_moe.runtime")


@pytest.mark.parametrize("t", [1, 8, 128])
@pytest.mark.parametrize("start", [0, 224])
def test_routes(t, start):
    em = torch.full((256,), -1, dtype=torch.int32)
    em[start : start + 32] = torch.arange(32, dtype=torch.int32)
    ids = torch.randint(0, 256, (t, 8), generator=torch.Generator().manual_seed(t))
    ids[:, :3] = start  # skew and repeated routes
    ids[0, -1] = -1
    lid, pos, cnt = runtime.local_routes(ids, em)
    expected = [0] * 32
    for i, raw in enumerate(ids.flatten().tolist()):
        e = int(em[raw]) if raw >= 0 else -1
        assert int(lid.flatten()[i]) == e
        if e >= 0:
            assert int(pos.flatten()[i]) == expected[e]
            expected[e] += 1
    assert cnt.tolist() == expected
    assert torch.equal(runtime.local_routes(lid)[2], cnt)


def make_norms():
    pre = norms.MuseSpark1xRMSNorm(32, eps=1e-5, zero_centered_gamma=True)
    post = norms.MuseSpark1xRMSNormWithInputScale(32, eps=1e-8, zero_centered_gamma=True)
    return pre, post


def test_exact_effective_weight():
    """The kernels receive stock's BF16 effective norm weight unchanged, for any value."""
    pre, post = make_norms()
    for n in (pre, post):
        n.gain_center = norms.calculate_gain_center("inverse_sqrt_residual_index", 2)
        n.weight.data.copy_(torch.linspace(-0.05, 0.05, 32))
        w = runtime.exact_weight(n)
        assert w.dtype == torch.bfloat16
        assert torch.equal(w, n._get_effective_weight().bfloat16())
        n.precompute_effective_weight()
        assert torch.equal(w, runtime.exact_weight(n))
    # Values that a 1 + w encoding could not represent exactly are passed through unchanged.
    pre.weight.data.fill_(-0.6)
    object.__setattr__(pre, "_frozen_weight", None)
    assert torch.equal(runtime.exact_weight(pre), pre._get_effective_weight().bfloat16())


def test_static_norm_contract():
    pre, post = make_norms()
    assert runtime.norm_contract(pre, post)
    assert runtime.norm_contract(None, post)
    post.eps = 1e-6
    assert not runtime.norm_contract(pre, post)
    post.eps = 1e-8
    pre.eps = 1e-6
    assert not runtime.norm_contract(pre, post)
    pre.eps = 1e-5
    assert not runtime.norm_contract(pre, pre)


def test_env_off_never_touches_weights(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.setenv("SGLANG_MUSE_W4A16_MOE_LAYERS", "0")
    assert envs.SGLANG_MUSE_W4A16_MOE_LAYERS.get() == 0
    assert not runtime.prepare_layer(types.SimpleNamespace())


def test_prep_eligibility_keeps_stock_for_bad_norms(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSE_W4A16_MOE_LAYERS", "-1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    def meta(shape, dtype):
        return types.SimpleNamespace(shape=shape, dtype=dtype, is_cuda=True, device="cuda")

    pre, post = make_norms()
    layer = types.SimpleNamespace(
        gate_up_proj=meta((32, 8192, 2048), torch.uint8),
        down_proj=meta((32, 4096, 2048), torch.uint8),
        gate_up_proj_weight_scale=meta((32, 8192, 256), torch.float8_e4m3fn),
        down_proj_weight_scale=meta((32, 4096, 256), torch.float8_e4m3fn),
        params_dtype=torch.bfloat16,
        top_k=8,
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        _w4a16_norms=(pre, post),
    )
    assert runtime.eligible(layer)
    for bad in (pre, norms.MuseSpark1xRMSNormWithInputScale(32, eps=1e-6)):
        layer._w4a16_norms = (pre, bad)
        assert not runtime.prepare_layer(layer)
        assert not hasattr(layer, "_w4a16_state")
        assert hasattr(layer, "gate_up_proj")


@pytest.mark.parametrize(
    "n,idx,expected",
    [
        (-1, None, True),
        (0, 1, False),
        (1, 1, True),
        (1, 2, False),
        (2, 2, True),
        (2, 3, False),
        (60, 60, True),
        (1, None, False),
        (60, 0, False),
        (61, 61, False),
    ],
)
def test_layer_limit(monkeypatch, n, idx, expected):
    monkeypatch.setenv("SGLANG_MUSE_W4A16_MOE_LAYERS", str(n))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    def meta(shape, dtype):
        return types.SimpleNamespace(shape=shape, dtype=dtype, is_cuda=True, device="cuda")

    layer = types.SimpleNamespace(
        gate_up_proj=meta((32, 8192, 2048), torch.uint8),
        down_proj=meta((32, 4096, 2048), torch.uint8),
        gate_up_proj_weight_scale=meta((32, 8192, 256), torch.float8_e4m3fn),
        down_proj_weight_scale=meta((32, 4096, 256), torch.float8_e4m3fn),
        params_dtype=torch.bfloat16,
        top_k=8,
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
    )
    if idx is not None:
        layer._w4a16_layer_idx = idx
    layer._w4a16_norms = make_norms()
    assert runtime.eligible(layer) is expected
    if not expected:
        assert not runtime.prepare_layer(layer)
        assert hasattr(layer, "gate_up_proj")


def test_missing_norm_metadata_keeps_marlin(monkeypatch):
    monkeypatch.setenv("SGLANG_MUSE_W4A16_MOE_LAYERS", "-1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    def meta(shape, dtype):
        return types.SimpleNamespace(shape=shape, dtype=dtype, is_cuda=True, device="cuda")

    layer = types.SimpleNamespace(
        gate_up_proj=meta((32, 8192, 2048), torch.uint8),
        down_proj=meta((32, 4096, 2048), torch.uint8),
        gate_up_proj_weight_scale=meta((32, 8192, 256), torch.float8_e4m3fn),
        down_proj_weight_scale=meta((32, 4096, 256), torch.float8_e4m3fn),
        params_dtype=torch.bfloat16,
        top_k=8,
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
    )
    assert runtime.eligible(layer) is False


def test_nvfp4_ordinal():
    assert [runtime._nvfp4_ordinal(i) for i in range(1, 61)] == list(range(60))


def test_build_at_load_and_finalize_before_capture(monkeypatch):
    build = importlib.import_module("sglang.srt.layers.muse_spark.w4a16_moe.build")
    calls = []
    mod = object()
    monkeypatch.setattr(build, "_module", lambda: calls.append("build") or mod)
    monkeypatch.setattr(runtime, "eligible", lambda layer: True)
    monkeypatch.setattr(runtime, "_repack", lambda *args: torch.empty(1))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(multi_processor_count=132),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    pre, post = make_norms()
    pre.weight.data.zero_()
    post.weight.data.zero_()
    layer = types.SimpleNamespace(_w4a16_norms=(pre, post), _expert_map=None)
    for name in (
        "gate_up_proj",
        "down_proj",
        "gate_up_proj_weight_scale",
        "down_proj_weight_scale",
        "gate_up_proj_weight_scale_2",
        "down_proj_weight_scale_2",
        "gate_up_proj_input_scale",
        "down_proj_input_scale",
    ):
        setattr(layer, name, torch.full((1, 1), 8, dtype=torch.uint8))
    assert runtime.prepare_layer(layer)
    assert calls == ["build"]
    assert layer._w4a16_state.mod is mod
    assert layer._w4a16_state.nsm == 132
    assert not hasattr(layer, "gate_up_proj")
    runtime.finalize_norms(layer)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    x = torch.empty((0, 4096), dtype=torch.bfloat16)
    routes = torch.empty((0, 8), dtype=torch.int64)
    assert runtime.forward(layer, x, routes.float(), routes, pre, post).shape == x.shape
    assert calls == ["build"]


def test_extension_initializes_once(monkeypatch):
    build = importlib.import_module("sglang.srt.layers.muse_spark.w4a16_moe.build")
    import torch.utils.cpp_extension as extension

    calls = []
    mod = types.SimpleNamespace(moe_init=lambda smem: calls.append(smem))
    monkeypatch.setattr(extension, "load_inline", lambda **kwargs: mod)
    build._module.cache_clear()
    try:
        assert build._module() is mod
        assert build._module() is mod
        assert calls == [runtime.SMEM]
    finally:
        build._module.cache_clear()
