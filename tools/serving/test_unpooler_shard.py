"""CPU tests: unpooler token-sharding (reduce-scatter -> GEMM -> all-gather) gate and numerics.

The real functions are extracted from muse_spark.py with ast and exec'd against stubs (no CUDA/sglang import).
"""

import ast
import os
import types
import unittest
from typing import Optional

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC = os.path.join(os.environ.get("SGLANG_TEST_CHECKOUT", os.path.join(ROOT, "serving/sglang")), "python/sglang/srt/models/muse_spark.py")
W = 8


def _load(names, g):
    tree = ast.parse(open(SRC).read())
    nodes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "_pad_rows" in nodes and {"_row_sharded_linear_gather", "_scatter_unpool_gather"}.intersection(names):
        names = ["_pad_rows", *names]
    mod = ast.Module(body=[nodes[n] for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), SRC, "exec"), g)
    return g


class FakeGroup:
    """Simulates rank `rank` of a W-rank group whose peers' partials are given; collectives are dim-0 only."""

    def __init__(self, partials, rank):
        self.partials, self.rank, self.world_size = partials, rank, len(partials)
        self.gathered = None  # filled by the harness for all_gather

    def reduce_scatter_along_dim(self, x, dim):
        assert dim == 0
        padded_partials = [F.pad(partial, (0, 0, 0, x.shape[0] - partial.shape[0])) for partial in self.partials]
        assert torch.equal(x, padded_partials[self.rank])
        total = sum(padded_partials)
        n = x.shape[0] // self.world_size
        return total[self.rank * n:(self.rank + 1) * n].contiguous()

    def all_gather(self, local, dim):
        assert dim == 0
        return torch.cat(self.gathered_parts, dim=0)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.st = types.SimpleNamespace(env=True, world=W, proj=True)
        envs = types.SimpleNamespace(SGLANG_MUSE_SHARD_UNPOOLER=types.SimpleNamespace(get=lambda: self.st.env))
        g = {"Optional": Optional, "ForwardBatch": object, "envs": envs, "torch": torch,
             "get_parallel": lambda: types.SimpleNamespace(tp_group=types.SimpleNamespace(world_size=self.st.world)),
             "_SHARD_UNPOOLER": None}
        _load(["_shard_unpooler_env", "_shard_unpool_enabled"], g)
        self.g = g
        self.fn = g["_shard_unpool_enabled"]
        w = types.SimpleNamespace(dtype=torch.bfloat16)
        self.me = types.SimpleNamespace(post_expert_proj=types.SimpleNamespace(weight=w), moe_tp_size=1,
                                        _shard_proj_enabled=lambda n, fb: self.st.proj)

    def call(self, T=16384):
        return self.fn(self.me, T, object())

    def test_true(self):
        self.assertTrue(self.call())

    def test_env_off(self):
        self.st.env = False
        self.assertFalse(self.call())

    def test_pooler_gate_false(self):
        self.st.proj = False
        self.assertFalse(self.call())

    def test_no_unpooler_or_moe_tp(self):
        self.me.post_expert_proj = None
        self.assertFalse(self.call())
        self.setUp()
        self.me.moe_tp_size = 2
        self.assertFalse(self.call())

    def test_dtype(self):
        self.me.post_expert_proj.weight.dtype = torch.float16
        self.assertFalse(self.call())

    def test_non_divisible_uses_sharded_path(self):
        self.assertTrue(self.call(16384 - 3))
        self.assertTrue(self.call(16384 - 8))

    def test_env_read_once(self):
        reads = []
        self.g["envs"].SGLANG_MUSE_SHARD_UNPOOLER.get = lambda: reads.append(1) or True
        for _ in range(4):
            self.call()
        self.assertEqual(len(reads), 1)


class NumericsTest(unittest.TestCase):
    def _run(self, T, dtype):
        torch.manual_seed(0)
        partials = [torch.randn(T, 64, dtype=dtype) for _ in range(W)]
        weight = torch.randn(96, 64, dtype=dtype)
        g = {"torch": torch, "bf16_gemm_dispatch": lambda x, w, b: F.linear(x, w),
             "get_parallel": lambda: types.SimpleNamespace(tp_group=self.group)}
        _load(["_pad_rows", "_scatter_unpool_gather"], g)
        fn = g["_scatter_unpool_gather"]
        # every rank computes its slice; all-gather concatenates them in rank order
        locals_ = []
        for r in range(W):
            grp = FakeGroup(partials, r)
            locals_.append(F.linear(grp.reduce_scatter_along_dim(partials[r], 0), weight))
        outs = []
        for r in range(W):
            self.group = FakeGroup(partials, r)
            self.group.gathered_parts = locals_
            outs.append(fn(partials[r], weight))
        ref = F.linear(sum(partials), weight)
        for o in outs:
            self.assertEqual(o.shape, ref.shape)
            # per-row GEMM is independent of the row slice; allow only fp reassociation noise
            torch.testing.assert_close(o, ref, rtol=1e-5, atol=1e-5)
            self.assertTrue(torch.equal(o, outs[0]))

    def test_fp32(self):
        for T in (8, 64, 8192):
            self._run(T, torch.float32)



if __name__ == "__main__":
    unittest.main()
