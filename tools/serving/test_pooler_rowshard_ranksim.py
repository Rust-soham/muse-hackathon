"""CPU rank-order and downstream contract simulation of the real row pooler."""
import ast
import types
import pytest
import torch
import torch.nn.functional as F
from test_unpooler_shard import _load
from test_pooler_shard_gate import SRC, _load_gate, _mk_self, Mode

HAS_ROW = '_row_sharded_linear_gather' in open(SRC).read()

@pytest.mark.skipif(not HAS_ROW, reason='column fallback has no row pooler')
@pytest.mark.parametrize('T', [8, 16, 8192, 8200, 16384])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_ranks(T, dtype):
    # Small integer products are exactly representable in both dtypes, so
    # equality tests rank ordering without tolerance hiding a permutation.
    gen = torch.Generator().manual_seed(T)
    x = torch.randint(-2, 3, (T, 16), generator=gen).to(dtype)
    w = torch.randint(-2, 3, (24, 16), generator=gen).to(dtype)
    expected = x @ w.T
    pieces = [F.linear(v, w) for v in x.chunk(8)]
    for rank in range(8):
        def gather(local, dim):
            assert dim == 0
            assert torch.equal(local, pieces[rank])
            return torch.cat(pieces, dim=0)
        group = types.SimpleNamespace(world_size=8, rank_in_group=rank, all_gather=gather)
        g = {'torch': torch, 'bf16_gemm_dispatch': F.linear,
             'get_parallel': lambda: types.SimpleNamespace(tp_group=group)}
        _load(['_pad_rows', '_row_sharded_linear_gather'], g)
        got = g['_row_sharded_linear_gather'](x, w)
        assert torch.equal(got, expected)
        assert (got.dtype, got.shape, got.is_contiguous()) == (expected.dtype, expected.shape, expected.is_contiguous())

@pytest.mark.skipif(not HAS_ROW, reason='column fallback has no row pooler')
def test_rank_gate_consensus():
    tree = ast.parse(open(SRC).read())
    forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_forward_tp')
    gate = next(n.value for n in forward.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'shard_proj' for t in n.targets))
    expr = compile(ast.Expression(gate), SRC, 'eval')
    for T, mode, sp, threshold, expected in [(t, 'extend', False, 8, True) for t in [8,16,8191,8192,8193,8200,16384]] + [(16,'extend',False,1024,False)] + [(8192,m,False,8,False) for m in ['decode','verify','mixed']] + [(8192,'extend',True,8,False)]:
        decisions=[]
        for rank in range(8):
            fn,g,st = _load_gate(); st.min_tokens=threshold; st.sp_active=sp
            me=_mk_self(); me._shard_proj_enabled=lambda t,fb: fn(me,t,fb)
            g['get_parallel']=lambda: types.SimpleNamespace(tp_group=types.SimpleNamespace(world_size=8,rank_in_group=rank))
            decisions.append(eval(expr,g,{'self':me,'hidden_states':types.SimpleNamespace(shape=(T,16)), 'forward_batch':types.SimpleNamespace(forward_mode=Mode(mode))}))
        assert decisions == [expected]*8
