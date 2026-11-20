"""CPU check of the real pooler row-shard helper with a fake eight-rank group."""
import types

import pytest
import torch
import torch.nn.functional as F

from test_unpooler_shard import _load


@pytest.mark.parametrize("T", [8, 64, 8192])
def test_row_gather(T):
    torch.manual_seed(0)
    x, weight = torch.randn(T, 64), torch.randn(96, 64)
    rows = T // 8
    locals_ = [F.linear(x[r * rows:(r + 1) * rows], weight) for r in range(8)]
    for rank in range(8):
        calls = []
        def dispatch(local_x, w, bias):
            assert w is weight and bias is None
            torch.testing.assert_close(local_x, x[rank * rows:(rank + 1) * rows])
            calls.append(True)
            return F.linear(local_x, w)
        def gather(local, dim):
            assert dim == 0
            torch.testing.assert_close(local, locals_[rank])
            return torch.cat(locals_, dim=0)
        group = types.SimpleNamespace(world_size=8, rank_in_group=rank, all_gather=gather)
        g = {"torch": torch, "bf16_gemm_dispatch": dispatch,
             "get_parallel": lambda: types.SimpleNamespace(tp_group=group)}
        _load(["_pad_rows", "_row_sharded_linear_gather"], g)
        got = g["_row_sharded_linear_gather"](x, weight)
        assert calls == [True]
        torch.testing.assert_close(got, F.linear(x, weight), rtol=1e-5, atol=1e-5)
