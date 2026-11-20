"""CPU rank simulation of the Muse Spark projection helpers and gates."""

import ast
import functools
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from test_pooler_shard_gate import Mode, _load_gate, _mk_self
from test_unpooler_shard import SRC, W, _load

PREFILL_TOKENS = (1024, 1025, 1031, 3012, 8191, 8192, 16384)
DECODE_TOKENS = (1, 8, 32, 64)


def _values(generator, rows, columns):
    # Integer products and sums allow exact comparisons across GEMM shapes.
    return torch.randint(-2, 3, (rows, columns), generator=generator).float()


def _check_output(actual, expected):
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", PREFILL_TOKENS)
def test_prefill_rank_numerics(tokens):
    generator = torch.Generator().manual_seed(tokens)
    x = _values(generator, tokens, 16)
    pre_weight = _values(generator, 24, 16)
    post_weight = _values(generator, 16, 24)
    partials = [_values(generator, tokens, 24) for _ in range(W)]
    pool = F.linear(x, pre_weight)
    unpool = F.linear(sum(partials), post_weight)
    pad = (-tokens) % W
    padded_partials = [F.pad(partial, (0, 0, 0, pad)) for partial in partials]
    reduced_chunks = sum(padded_partials).chunk(W)
    pool_chunks = F.pad(pool, (0, 0, 0, pad)).chunk(W)
    unpool_chunks = F.pad(unpool, (0, 0, 0, pad)).chunk(W)

    def gather(chunks, local, dim):
        assert dim == 0
        _check_output(local, chunks[rank])
        return torch.cat(chunks, dim=0)

    def scatter(padded, dim):
        assert dim == 0
        _check_output(padded, padded_partials[rank])
        return reduced_chunks[rank]

    group = SimpleNamespace(world_size=W, reduce_scatter_along_dim=scatter)
    g = _load(
        ["_row_sharded_linear_gather", "_scatter_unpool_gather"],
        {"torch": torch, "F": F, "_SHARD_DECODE_MAX_TOKENS": None, "bf16_gemm_dispatch": F.linear,
         "get_parallel": lambda: SimpleNamespace(tp_group=group)},
    )
    for rank in range(W):
        group.rank_in_group = rank
        group.all_gather = functools.partial(gather, pool_chunks)
        _check_output(g["_row_sharded_linear_gather"](x, pre_weight), pool)
        group.all_gather = functools.partial(gather, unpool_chunks)
        _check_output(g["_scatter_unpool_gather"](partials[rank], post_weight), unpool)


@pytest.mark.parametrize("tokens", DECODE_TOKENS)
def test_decode_rank_numerics(tokens):
    generator = torch.Generator().manual_seed(tokens)
    x = _values(generator, tokens, 16)
    pre_weight = _values(generator, 24, 16)
    post_weight = _values(generator, 16, 24)
    partials = [_values(generator, tokens, 24) for _ in range(W)]
    group = SimpleNamespace(world_size=W, ca_comm=None)
    g = _load(
        ["_column_sharded_linear_reduce", "_reduce_and_unpool"],
        {"torch": torch, "F": F, "_SHARD_DECODE_MAX_TOKENS": None, "bf16_gemm_dispatch": F.linear,
         "gemm_ag_supported": lambda comm, x, weight: False,
         "get_parallel": lambda: SimpleNamespace(tp_group=group),
         "get_forward": lambda: SimpleNamespace(sp_active=False), "_is_cpu": False},
    )
    model = SimpleNamespace(
        tp_size=W, moe_tp_size=1, post_expert_proj=SimpleNamespace(weight=post_weight)
    )

    for data, weight in [(x, pre_weight), (sum(partials), post_weight)]:
        expected = F.linear(data, weight)
        columns = weight.shape[0] // W
        shards = []
        for peer in range(W):
            start = peer * columns
            shard = torch.zeros_like(expected)
            shard[:, start : start + columns] = F.linear(
                data, weight[start : start + columns]
            )
            shards.append(shard)

        def reduce(local):
            if local is partials[rank]:
                calls.append("expert")
                return sum(partials)
            calls.append("projection")
            _check_output(local, shards[rank])
            return sum(shards)

        g["tensor_model_parallel_all_reduce"] = reduce
        for rank in range(W):
            group.rank_in_group = rank
            calls = []
            if weight is pre_weight:
                actual = g["_column_sharded_linear_reduce"](data, weight)
                assert calls == ["projection"]
            else:
                actual = g["_reduce_and_unpool"](model, partials[rank], shard_decode=True)
                assert calls == ["expert", "projection"]
                calls.clear()
                _check_output(
                    g["_reduce_and_unpool"](
                        model, data, already_reduced=True, shard_decode=True
                    ),
                    expected,
                )
                assert calls == ["projection"]
            _check_output(actual, expected)


def _knob_default(name):
    env = Path(SRC).parents[1] / "environ.py"
    tree = ast.parse(env.read_text())
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return ast.literal_eval(assignment.value.args[0])


def test_prefill_gates():
    pool_gate, g, state = _load_gate()
    state.min_tokens = _knob_default("SGLANG_MUSE_SHARD_POOLER_MIN_TOKENS")
    assert state.min_tokens == 1024
    model = _mk_self()
    model.moe_tp_size = 1
    model._shard_proj_enabled = functools.partial(pool_gate, model)
    g["_shard_unpooler_env"] = lambda: True
    unpool_gate = _load(["_shard_unpool_enabled"], g)["_shard_unpool_enabled"]
    cases = [(tokens, "extend", False, True) for tokens in PREFILL_TOKENS]
    cases += [(1024, mode, False, False) for mode in ("decode", "verify", "mixed")]
    cases += [(1024, "extend", True, False), (1023, "extend", False, False)]
    for tokens, mode, sp_active, expected in cases:
        state.sp_active = sp_active
        batch = SimpleNamespace(forward_mode=Mode(mode))
        assert pool_gate(model, tokens, batch) is expected
        assert unpool_gate(model, tokens, batch) is expected

    g["_SHARD_MIN_TOKENS"] = None
    state.min_tokens = 3012
    batch = SimpleNamespace(forward_mode=Mode("extend"))
    assert not pool_gate(model, 3011, batch)
    assert not unpool_gate(model, 3011, batch)
    assert pool_gate(model, 3012, batch)
    assert unpool_gate(model, 3012, batch)


def test_decode_gates():
    _, g, state = _load_gate()
    knob = "SGLANG_MUSE_SHARD_DECODE_MAX_TOKENS"
    limit = _knob_default(knob)
    assert limit == 64
    reads = []

    def read_limit():
        reads.append(limit)
        return limit

    setattr(g["envs"], knob, SimpleNamespace(get=read_limit))
    g["functools"] = functools
    g["_SHARD_DECODE_MAX_TOKENS"] = None
    _load(["_shard_decode_max_tokens", "_shard_decode_enabled"], g)
    model = _mk_self()
    gate = functools.partial(g["_shard_decode_enabled"], model)

    for mode in ("decode", "verify"):
        batch = SimpleNamespace(forward_mode=Mode(mode))
        for tokens in DECODE_TOKENS:
            assert gate(tokens, batch)
        for tokens in (0, 65, 128, 256):
            assert not gate(tokens, batch)
    for mode in ("extend", "mixed"):
        assert not gate(8, SimpleNamespace(forward_mode=Mode(mode)))
    assert not gate(8, None)
    batch = SimpleNamespace(forward_mode=Mode("decode"))
    state.sp_active = True
    assert not gate(8, batch)
    state.sp_active = False
    state.world = 4
    assert not gate(8, batch)
    state.world = W
    state.cap = 8
    assert not gate(8, batch)
    state.cap = 9
    g["_is_cpu"] = True
    assert not gate(8, batch)
    g["_is_cpu"] = False
    original = model.post_expert_proj
    model.post_expert_proj = None
    assert not gate(8, batch)
    model.post_expert_proj = SimpleNamespace(
        weight=SimpleNamespace(dtype=torch.bfloat16, shape=(4097, 8192))
    )
    assert not gate(8, batch)
    model.post_expert_proj.weight.dtype = torch.float16
    model.post_expert_proj.weight.shape = (4096, 8192)
    assert not gate(8, batch)
    model.post_expert_proj = original
    assert reads == [64]
    limit = 0
    g["_SHARD_DECODE_MAX_TOKENS"] = None
    assert not gate(8, batch)
