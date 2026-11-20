"""Restricted Hopper expert backend; output is the BF16 partial before TP reduce.

Norm validation and exact effective-weight conversion precede the first launch.
Routing and all per-call allocations are graph safe; no device values reach Python.
"""

import logging
from dataclasses import dataclass

import torch

from .packing import _repack

PRE_EPS = 1e-5
POST_EPS = 1e-8
SMEM = 227 * 1024
# Peak scratch at 16384 tokens is ~3.6 GB, close to stock Marlin's ~3.4 GB at the same size.
_SLICE_TOKENS = 16384

logger = logging.getLogger(__name__)
_LOGGED = False


def norm_contract(pre, post):
    from sglang.srt.layers.muse_spark.muse_spark_layers import (
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale,
    )

    return (
        type(post) is MuseSpark1xRMSNormWithInputScale
        and post.eps == POST_EPS
        and (pre is None or (type(pre) is MuseSpark1xRMSNorm and pre.eps == PRE_EPS))
    )


def exact_weight(norm, device=None):
    """The stock BF16 effective norm weight, which the kernels read as-is."""
    effective = norm._get_effective_weight()
    if effective is None:
        effective = torch.ones(norm.hidden_size, device=device)
    return effective.detach().to(device=device, dtype=torch.bfloat16).contiguous()


def local_routes(ids, expert_map=None):
    """Flatten token/slot order; positions are exclusive per-expert prefix sums."""
    ids = ids.to(torch.int64)
    if expert_map is not None:
        mapped = expert_map[ids.clamp_min(0)]
        lid = torch.where(ids >= 0, mapped, -1)
    else:
        lid = ids
    lid = torch.where((lid >= 0) & (lid < 32), lid, -1).to(torch.int32).contiguous()
    # Expert-major [32, routes]: the scan runs along the contiguous last dim. A scan over dim 0
    # of the route-major layout is near-serial on CUDA and costs ~0.2 us per route.
    hit = (torch.arange(32, device=ids.device).view(32, 1) == lid.view(1, -1)).to(torch.int32)
    prefix = hit.cumsum(1, dtype=torch.int32) - 1
    pos = (prefix * hit).sum(0, dtype=torch.int32).reshape_as(lid)
    counts = hit.sum(1, dtype=torch.int32)
    return lid, pos, counts


@dataclass
class State:
    w13: torch.Tensor
    w2: torch.Tensor
    gs13: torch.Tensor
    gs2: torch.Tensor
    pre: object = None
    post: object = None
    pre_weight: object = None
    post_weight: object = None
    mod: object = None
    nsm: int = 0


def _nvfp4_ordinal(idx):
    """Muse Spark layers 0 and 61 are BF16; NVFP4 layers 1..60 start at zero."""
    return idx - 1


def eligible(layer):
    """Select by model index so ranks agree regardless of checkpoint load order."""
    from sglang.srt.environ import envs

    n = envs.SGLANG_MUSE_W4A16_MOE_LAYERS.get()
    if n == 0:
        return False
    idx = getattr(layer, "_w4a16_layer_idx", None)
    if n > 0 and (idx is None or not 1 <= idx <= 60 or _nvfp4_ordinal(idx) >= n):
        return False
    w = layer.gate_up_proj
    if not w.is_cuda or torch.cuda.get_device_capability(w.device) != (9, 0):
        return False
    if (
        layer.params_dtype != torch.bfloat16
        or layer.top_k != 8
        or layer.quant_config.group_size != 16
        or not layer.moe_runner_config.is_gated
        or tuple(w.shape) != (32, 8192, 2048)
        or w.dtype != torch.uint8
        or tuple(layer.down_proj.shape) != (32, 4096, 2048)
        or layer.down_proj.dtype != torch.uint8
    ):
        return False
    for name, shape in (
        ("gate_up_proj_weight_scale", (32, 8192, 256)),
        ("down_proj_weight_scale", (32, 4096, 256)),
    ):
        s = getattr(layer, name)
        if tuple(s.shape) != shape or s.dtype != torch.float8_e4m3fn:
            return False
    # The parent model binds its norms before checkpoint loading; without them the norm
    # contract cannot be checked before the native weights are released, so keep Marlin.
    norms = getattr(layer, "_w4a16_norms", None)
    return norms is not None and norm_contract(*norms)


@torch.no_grad()
def prepare_layer(layer):
    """Replace native weights with W4A16 only, keeping repack memory neutral.

    Build during loading so the first forward cannot compile inside a graph.
    """
    if not eligible(layer):
        return False
    # The kernel's scale decode is exact only for positive normal E4M3 bytes. Otherwise this
    # rank keeps Marlin for this layer; both backends return the same pre-all-reduce partial,
    # so ranks may differ in backend without changing collectives.
    for name in ("gate_up_proj_weight_scale", "down_proj_weight_scale"):
        s = getattr(layer, name).view(torch.uint8)
        if not bool(((s >= 8) & (s <= 126)).all()):
            logger.info("W4A16 MoE: layer keeps Marlin (E4M3 block scale outside 0x08..0x7E)")
            return False
    gs13 = layer.gate_up_proj_weight_scale_2[:, 0].detach().float().contiguous()
    gs2 = layer.down_proj_weight_scale_2.detach().float().contiguous()
    packed = []
    for wn, sn, gated in (
        ("gate_up_proj", "gate_up_proj_weight_scale", True),
        ("down_proj", "down_proj_weight_scale", False),
    ):
        # _repack bounds transient expansion to one expert, then releases it.
        packed.append(_repack(getattr(layer, wn), getattr(layer, sn), gated))
        delattr(layer, wn)
        delattr(layer, sn)
    for name in (
        "gate_up_proj_weight_scale_2",
        "down_proj_weight_scale_2",
        "gate_up_proj_input_scale",
        "down_proj_input_scale",
    ):
        delattr(layer, name)
    # Native tensors and per-expert scratch are dead; release unused segments
    # before the loader allocates the next layer or measures device free memory.
    torch.cuda.empty_cache()
    from .build import _module

    mod = _module()
    layer._w4a16_state = State(
        *packed,
        gs13,
        gs2,
        mod=mod,
        nsm=torch.cuda.get_device_properties(gs13.device).multi_processor_count,
    )
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info("W4A16 MoE: NVFP4 experts use the fused W4A16 kernel instead of Marlin")
    return True


def _norms(st, pre, post):
    """Convert loaded norms outside capture; extension initialization is load-only."""
    if st.post_weight is not None and st.pre is pre and st.post is post:
        return
    if not norm_contract(pre, post):
        raise ValueError(
            "W4A16 MoE requires norm-first MuseSpark1xRMSNorm eps=1e-5 (or no pre "
            "norm), and weight-first MuseSpark1xRMSNormWithInputScale eps=1e-8; "
            "disable SGLANG_MUSE_W4A16_MOE_LAYERS before loading"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("W4A16 MoE needs one warm-up forward before CUDA graph capture")
    st.pre_weight = (
        exact_weight(pre, st.w13.device)
        if pre is not None
        else torch.empty(0, dtype=torch.bfloat16, device=st.w13.device)
    )
    st.post_weight = exact_weight(post, st.w13.device)
    st.pre, st.post = pre, post


def finalize_norms(layer):
    """Finalize after norm loading so graph capture never needs torch conversion."""
    st = getattr(layer, "_w4a16_state", None)
    if st is not None and hasattr(layer, "_w4a16_norms"):
        _norms(st, *layer._w4a16_norms)


def _slice(st, x, scores, ids, expert_map, normalize):
    t = x.shape[0]
    dev = x.device
    lid, pos, cnt = local_routes(ids, expert_map)
    maxrows, maxtiles = t * 8 + 32 * 256, t * 8 // 32 + 64

    def ints(*shape):
        return torch.empty(shape, dtype=torch.int32, device=dev)

    offsets, rows, tiles, ntiles = ints(32), ints(32), ints(maxtiles, 4), ints(1)
    xt = torch.empty((maxrows, 4096), dtype=torch.bfloat16, device=dev)
    inter, y = torch.empty_like(xt), torch.empty_like(xt)
    rrow = ints(t, 8)
    empty = ints(0)
    fine = int(t >= 512) << 8
    st.mod.moe_gather_plan(
        x,
        st.pre_weight,
        lid,
        pos,
        cnt,
        xt,
        rrow,
        offsets,
        tiles,
        ntiles,
        rows,
        1 | fine,
        empty,
        empty,
        normalize,
    )
    counters = torch.zeros((1 + maxtiles,), dtype=torch.int32, device=dev)
    empty_f = torch.empty(0, dtype=torch.float32, device=dev)
    empty_b = torch.empty(0, dtype=torch.bfloat16, device=dev)
    st.mod.moe_fused(
        st.w13,
        st.w2,
        xt,
        inter,
        y,
        tiles,
        ntiles,
        st.gs13,
        st.gs2,
        st.nsm,
        SMEM,
        counters[:1],
        counters[1:],
        empty,
        fine,
        empty,
        empty,
        rrow,
        lid,
        empty_f,
        st.post_weight,
        empty_b,
    )
    out = torch.empty_like(x)
    st.mod.moe_combine(y, rrow, lid, scores.float().contiguous(), st.post_weight, out)
    return out


@torch.no_grad()
def forward(layer, x, scores, ids, pre, post):
    st = layer._w4a16_state
    if (
        x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] != 4096
        or ids.shape != (x.shape[0], 8)
        or scores.shape != ids.shape
    ):
        raise ValueError("W4A16 MoE expects BF16 [T,4096] input and [T,8] routes")
    _norms(st, pre, post)
    if not x.shape[0]:
        return torch.empty_like(x)
    outputs = [
        _slice(
            st,
            x[a : a + _SLICE_TOKENS].contiguous(),
            scores[a : a + _SLICE_TOKENS],
            ids[a : a + _SLICE_TOKENS],
            layer._expert_map,
            pre is not None,
        )
        for a in range(0, x.shape[0], _SLICE_TOKENS)
    ]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
