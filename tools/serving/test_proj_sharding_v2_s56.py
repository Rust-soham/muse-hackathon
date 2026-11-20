"""CPU rank simulations for Muse Spark projection sharding."""

import ast
import os
import types
from typing import Optional

import pytest
import torch
import torch.nn.functional as F


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CHECKOUT = os.environ.get(
    "SGLANG_TEST_CHECKOUT", os.path.join(ROOT, "serving", "sglang")
)
MODEL_SRC = os.path.join(
    CHECKOUT, "python", "sglang", "srt", "models", "muse_spark.py"
)
WORLD_SIZE = 8


def _load_functions(names, globals_):
    tree = ast.parse(open(MODEL_SRC).read())
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    module = ast.Module(body=[functions[name] for name in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), MODEL_SRC, "exec"), globals_)
    return globals_


def _integer_tensor(shape, generator):
    return torch.randint(-2, 3, shape, generator=generator).float()


@pytest.mark.parametrize("num_tokens", [1024, 1025, 1031, 3012, 8191, 8192, 16384])
def test_padded_row_sharded_pooler_matches_replicated(num_tokens):
    generator = torch.Generator().manual_seed(num_tokens)
    x = _integer_tensor((num_tokens, 16), generator)
    weight = _integer_tensor((24, 16), generator)
    padded = F.pad(x, (0, 0, 0, (-num_tokens) % WORLD_SIZE))
    rows = padded.shape[0] // WORLD_SIZE
    local_outputs = [
        F.linear(padded[rank * rows : (rank + 1) * rows], weight)
        for rank in range(WORLD_SIZE)
    ]
    expected = F.linear(x, weight)

    for rank in range(WORLD_SIZE):
        group = types.SimpleNamespace(
            world_size=WORLD_SIZE,
            rank_in_group=rank,
            all_gather=lambda local, dim, rank=rank: _gather_local(
                local, dim, rank, local_outputs
            ),
        )
        globals_ = {
            "torch": torch,
            "F": F,
            "bf16_gemm_dispatch": lambda local_x, w, bias: F.linear(local_x, w),
            "get_parallel": lambda group=group: types.SimpleNamespace(tp_group=group),
        }
        _load_functions(["_pad_rows", "_row_sharded_linear_gather"], globals_)
        output = globals_["_row_sharded_linear_gather"](x, weight)
        assert torch.equal(output, expected)
        assert output.is_contiguous()


def _gather_local(local, dim, rank, local_outputs):
    assert dim == 0
    assert torch.equal(local, local_outputs[rank])
    return torch.cat(local_outputs, dim=0)


class _ReduceScatterGroup:
    def __init__(self, padded_partials, local_outputs, rank):
        self.world_size = len(padded_partials)
        self.padded_partials = padded_partials
        self.local_outputs = local_outputs
        self.rank = rank

    def reduce_scatter_along_dim(self, partial, dim):
        assert dim == 0
        assert torch.equal(partial, self.padded_partials[self.rank])
        total = sum(self.padded_partials)
        rows = total.shape[0] // self.world_size
        return total[self.rank * rows : (self.rank + 1) * rows].contiguous()

    def all_gather(self, local, dim):
        return _gather_local(local, dim, self.rank, self.local_outputs)


@pytest.mark.parametrize("num_tokens", [1024, 1025, 1031, 3012, 8191, 8192, 16384])
def test_padded_row_sharded_unpooler_matches_replicated(num_tokens):
    generator = torch.Generator().manual_seed(num_tokens + 1)
    partials = [
        _integer_tensor((num_tokens, 16), generator) for _ in range(WORLD_SIZE)
    ]
    weight = _integer_tensor((24, 16), generator)
    pad = (-num_tokens) % WORLD_SIZE
    padded_partials = [F.pad(partial, (0, 0, 0, pad)) for partial in partials]
    total = sum(padded_partials)
    rows = total.shape[0] // WORLD_SIZE
    local_outputs = [
        F.linear(total[rank * rows : (rank + 1) * rows], weight)
        for rank in range(WORLD_SIZE)
    ]
    expected = F.linear(sum(partials), weight)

    for rank in range(WORLD_SIZE):
        group = _ReduceScatterGroup(padded_partials, local_outputs, rank)
        globals_ = {
            "torch": torch,
            "F": F,
            "bf16_gemm_dispatch": lambda local_x, w, bias: F.linear(local_x, w),
            "get_parallel": lambda group=group: types.SimpleNamespace(tp_group=group),
        }
        _load_functions(["_pad_rows", "_scatter_unpool_gather"], globals_)
        output = globals_["_scatter_unpool_gather"](partials[rank], weight)
        assert torch.equal(output, expected)
        assert output.is_contiguous()


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "decode"

    def is_target_verify(self):
        return self.name == "target_verify"

    def is_extend_without_speculative(self):
        return self.name in ("extend", "mixed")

    def is_mixed(self):
        return self.name == "mixed"


def _load_prefill_gate(*, threshold=1024, cpu=False, sp_active=False):
    state = types.SimpleNamespace(threshold=threshold)
    envs = types.SimpleNamespace(
        SGLANG_MUSE_SHARD_POOLER_MIN_TOKENS=types.SimpleNamespace(
            get=lambda: state.threshold
        )
    )
    globals_ = {
        "Optional": Optional,
        "ForwardBatch": object,
        "envs": envs,
        "torch": types.SimpleNamespace(
            bfloat16=torch.bfloat16,
            cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0)),
        ),
        "get_forward": lambda: types.SimpleNamespace(sp_active=sp_active),
        "get_parallel": lambda: types.SimpleNamespace(
            tp_group=types.SimpleNamespace(world_size=WORLD_SIZE)
        ),
        "_is_cpu": cpu,
        "_SHARD_MIN_TOKENS": None,
    }
    _load_functions(["_shard_min_tokens", "_shard_proj_enabled"], globals_)
    return globals_["_shard_proj_enabled"], globals_


def _projection_owner(gate):
    weight = types.SimpleNamespace(
        shape=(4096, 8192), dtype=torch.bfloat16
    )
    owner = types.SimpleNamespace(
        pre_expert_proj=types.SimpleNamespace(weight=weight), tp_size=WORLD_SIZE
    )
    owner._shard_proj_enabled = lambda tokens, batch: gate(owner, tokens, batch)
    return owner


@pytest.mark.parametrize("num_tokens", [1024, 1025, 1031, 3012, 8191, 8192, 16384])
def test_prefill_gate_accepts_threshold_and_padded_sizes(num_tokens):
    gate, _ = _load_prefill_gate()
    owner = _projection_owner(gate)
    batch = types.SimpleNamespace(forward_mode=_Mode("extend"))
    assert gate(owner, num_tokens, batch)


@pytest.mark.parametrize("mode", ["decode", "target_verify", "mixed"])
def test_prefill_gate_rejects_non_prefill_modes(mode):
    gate, _ = _load_prefill_gate()
    owner = _projection_owner(gate)
    batch = types.SimpleNamespace(forward_mode=_Mode(mode))
    assert not gate(owner, 8192, batch)


def test_prefill_gate_respects_threshold_sp_and_cpu():
    batch = types.SimpleNamespace(forward_mode=_Mode("extend"))
    gate, _ = _load_prefill_gate(threshold=1024)
    assert not gate(_projection_owner(gate), 1023, batch)
    gate, _ = _load_prefill_gate(threshold=0)
    assert not gate(_projection_owner(gate), 8192, batch)
    gate, _ = _load_prefill_gate(sp_active=True)
    assert not gate(_projection_owner(gate), 8192, batch)
    gate, _ = _load_prefill_gate(cpu=True)
    assert not gate(_projection_owner(gate), 8192, batch)


@pytest.mark.parametrize("projection", ["pooler", "unpooler"])
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 64])
def test_column_sharded_projection_matches_replicated(projection, num_tokens):
    generator = torch.Generator().manual_seed(num_tokens)
    input_width = 16
    output_width = 24 if projection == "pooler" else 32
    x = _integer_tensor((num_tokens, input_width), generator)
    weight = _integer_tensor((output_width, input_width), generator)
    rows = output_width // WORLD_SIZE
    rank_outputs = []
    for rank in range(WORLD_SIZE):
        rank_output = x.new_zeros((num_tokens, output_width))
        start = rank * rows
        rank_output[:, start : start + rows] = F.linear(
            x, weight[start : start + rows]
        )
        rank_outputs.append(rank_output)
    expected = F.linear(x, weight)

    for rank in range(WORLD_SIZE):
        group = types.SimpleNamespace(world_size=WORLD_SIZE, rank_in_group=rank)

        def all_reduce(local, rank=rank):
            assert torch.equal(local, rank_outputs[rank])
            return sum(rank_outputs)

        globals_ = {
            "torch": torch,
            "bf16_gemm_dispatch": lambda local_x, w, bias: F.linear(local_x, w),
            "tensor_model_parallel_all_reduce": all_reduce,
            "get_parallel": lambda group=group: types.SimpleNamespace(tp_group=group),
        }
        _load_functions(["_column_sharded_linear_reduce"], globals_)
        output = globals_["_column_sharded_linear_reduce"](x, weight)
        assert torch.equal(output, expected)
        assert output.is_contiguous()


def _load_decode_gate(*, max_tokens=64, cpu=False, sp_active=False):
    state = types.SimpleNamespace(max_tokens=max_tokens)
    envs = types.SimpleNamespace(
        SGLANG_MUSE_SHARD_DECODE_MAX_TOKENS=types.SimpleNamespace(
            get=lambda: state.max_tokens
        )
    )
    globals_ = {
        "Optional": Optional,
        "ForwardBatch": object,
        "envs": envs,
        "torch": types.SimpleNamespace(
            bfloat16=torch.bfloat16,
            cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0)),
        ),
        "get_forward": lambda: types.SimpleNamespace(sp_active=sp_active),
        "get_parallel": lambda: types.SimpleNamespace(
            tp_group=types.SimpleNamespace(world_size=WORLD_SIZE)
        ),
        "_is_cpu": cpu,
        "_SHARD_DECODE_MAX_TOKENS": None,
    }
    _load_functions(
        ["_shard_decode_max_tokens", "_shard_decode_enabled"], globals_
    )
    return globals_["_shard_decode_enabled"], globals_


def _decode_projection_owner(gate):
    pre_weight = types.SimpleNamespace(
        shape=(4096, 8192), dtype=torch.bfloat16
    )
    post_weight = types.SimpleNamespace(
        shape=(8192, 4096), dtype=torch.bfloat16
    )
    return types.SimpleNamespace(
        pre_expert_proj=types.SimpleNamespace(weight=pre_weight),
        post_expert_proj=types.SimpleNamespace(weight=post_weight),
        tp_size=WORLD_SIZE,
    )


@pytest.mark.parametrize("mode", ["decode", "target_verify"])
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 64])
def test_decode_gate_accepts_small_decode_and_verify(mode, num_tokens):
    gate, _ = _load_decode_gate()
    owner = _decode_projection_owner(gate)
    batch = types.SimpleNamespace(forward_mode=_Mode(mode))
    assert gate(owner, num_tokens, batch)


@pytest.mark.parametrize("num_tokens", [65, 128, 256])
def test_decode_gate_rejects_large_batches(num_tokens):
    gate, _ = _load_decode_gate()
    owner = _decode_projection_owner(gate)
    batch = types.SimpleNamespace(forward_mode=_Mode("decode"))
    assert not gate(owner, num_tokens, batch)


@pytest.mark.parametrize("mode", ["extend", "mixed"])
def test_decode_gate_rejects_prefill_modes(mode):
    gate, _ = _load_decode_gate()
    owner = _decode_projection_owner(gate)
    batch = types.SimpleNamespace(forward_mode=_Mode(mode))
    assert not gate(owner, 32, batch)


def test_decode_gate_rejects_sp_cpu_and_disabled_knob():
    batch = types.SimpleNamespace(forward_mode=_Mode("decode"))
    gate, _ = _load_decode_gate(sp_active=True)
    assert not gate(_decode_projection_owner(gate), 32, batch)
    gate, _ = _load_decode_gate(cpu=True)
    assert not gate(_decode_projection_owner(gate), 32, batch)
    gate, _ = _load_decode_gate(max_tokens=0)
    assert not gate(_decode_projection_owner(gate), 32, batch)
