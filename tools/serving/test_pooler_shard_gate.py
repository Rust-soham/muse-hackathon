"""CPU tests: MuseSpark1xMoE._shard_proj_enabled gate + the sharded pooler product (_row_sharded_linear_gather).

The gate is exercised from the real source: the method is extracted with ast and
exec'd against stub globals (no sglang/CUDA import needed). Run:
    .venv-cpu/bin/python -m pytest tools/serving/test_pooler_shard_gate.py
"""

import ast
import os
import types
import unittest

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC = os.path.join(os.environ.get("SGLANG_TEST_CHECKOUT", os.path.join(ROOT, "serving/sglang")), "python/sglang/srt/models/muse_spark.py")


def _load_gate():
    tree = ast.parse(open(SRC).read())
    nodes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "_shard_proj_enabled" not in nodes or "_shard_min_tokens" not in nodes:
        raise AssertionError("_shard_proj_enabled/_shard_min_tokens not found")
    mod = ast.Module(body=[nodes["_shard_min_tokens"], nodes["_shard_proj_enabled"]], type_ignores=[])
    state = types.SimpleNamespace(min_tokens=1024, capturing=False, sp_active=False,
                                  world=8, cap=9, is_cpu=False)
    envs = types.SimpleNamespace(
        SGLANG_MUSE_SHARD_POOLER_MIN_TOKENS=types.SimpleNamespace(
            get=lambda: state.min_tokens))
    cuda = types.SimpleNamespace(
        is_current_stream_capturing=lambda: state.capturing,
        get_device_capability=lambda: (state.cap, 0))
    tch = types.SimpleNamespace(cuda=cuda, bfloat16=torch.bfloat16)
    g = {
        "Optional": __import__("typing").Optional, "ForwardBatch": object,
        "envs": envs, "torch": tch, "get_forward": lambda: types.SimpleNamespace(sp_active=state.sp_active),
        "get_parallel": lambda: types.SimpleNamespace(tp_group=types.SimpleNamespace(world_size=state.world)),
    }
    g["_is_cpu"] = False
    g["_SHARD_MIN_TOKENS"] = None
    exec(compile(ast.fix_missing_locations(mod), SRC, "exec"), g)
    return g["_shard_proj_enabled"], g, state


class Mode:
    """Mirrors ForwardMode in model_executor/forward_batch_info.py: is_extend() is True for MIXED and TARGET_VERIFY."""

    def __init__(self, name):
        self.name = name

    def is_extend(self):
        return self.name in ("extend", "mixed", "verify", "split_prefill", "dllm_extend")

    def is_target_verify(self):
        return self.name == "verify"

    def is_extend_without_speculative(self):
        return self.is_extend() and not self.is_target_verify()

    def is_mixed(self):
        return self.name == "mixed"

    def is_decode(self):
        return self.name == "decode"


def _mk_self():
    w = types.SimpleNamespace(shape=(4096, 8192), dtype=torch.bfloat16)
    return types.SimpleNamespace(
        pre_expert_proj=types.SimpleNamespace(weight=w),
        post_expert_proj=types.SimpleNamespace(weight=w), tp_size=8)


class GateTest(unittest.TestCase):
    def test_env_read_once(self):
        reads = []
        self.g["envs"].SGLANG_MUSE_SHARD_POOLER_MIN_TOKENS.get = lambda: reads.append(1) or 1024
        for _ in range(5):
            self.call()
        self.assertEqual(len(reads), 1)

    def setUp(self):
        self.fn, self.g, self.st = _load_gate()
        self.me = _mk_self()

    def call(self, mode="extend", T=4096):
        fb = types.SimpleNamespace(forward_mode=Mode(mode))
        return self.fn(self.me, T, fb)

    def test_baseline_true(self):
        self.assertTrue(self.call("extend"))

    def test_false_for_mixed(self):
        # A mixed batch carries running decode rows next to the prefill tokens; they keep stock's GEMM.
        self.assertFalse(self.call("mixed"))

    def test_false_decode_and_verify(self):
        self.assertFalse(self.call("decode"))
        self.assertFalse(self.call("verify"))

    def test_still_true_under_capture(self):
        # Prefill replays captured graphs up to chunked_prefill_size (16,384 tokens), so a path that is off under
        # capture would never run. The gather-based path uses the unpooler collective, valid in graphs.
        self.st.capturing = True
        self.assertTrue(self.call())

    def test_false_when_min_tokens_zero(self):
        self.st.min_tokens = 0
        self.g["_SHARD_MIN_TOKENS"] = None  # the env value is cached on first use
        self.assertFalse(self.call())

    def test_false_below_threshold(self):
        self.assertFalse(self.call(T=self.st.min_tokens - 1))
        self.assertTrue(self.call(T=self.st.min_tokens))

    def test_other_guards(self):
        self.assertFalse(self.fn(self.me, 4096, None))
        self.st.sp_active = True
        self.assertFalse(self.call())
        self.st.sp_active = False
        self.st.cap = 8
        self.assertFalse(self.call())


    def test_non_divisible_uses_sharded_path(self):
        tree = ast.parse(open(SRC).read())
        forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_forward_tp")
        gate = next(n.value for n in forward.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "shard_proj" for t in n.targets))
        expr = compile(ast.Expression(gate), SRC, "eval")
        self.me._shard_proj_enabled = lambda T, fb: self.fn(self.me, T, fb)
        for mode, T, expected in [("extend", 8192, True), ("extend", 8193, True), ("decode", 8192, False), ("verify", 8192, False), ("mixed", 8192, False)]:
            self.assertEqual(eval(expr, self.g, {"self": self.me, "hidden_states": types.SimpleNamespace(shape=(T, 8192)), "forward_batch": types.SimpleNamespace(forward_mode=Mode(mode))}), expected)
        experts = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_forward_experts")
        branch = next(n for n in ast.walk(experts) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "shard_proj")
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "bf16_gemm_dispatch" for stmt in branch.orelse for n in ast.walk(stmt)))


if __name__ == "__main__":
    unittest.main()
