"""Lossless permutation of NVFP4 codes and FP8 scales; no requantization.

Extracted from muse_prefill_layer/submission.py (_codes, _CODE_POS, _repack).
Gate rows precede up rows, as in native HF weights and Marlin preparation.
"""
import torch

KS = 64
WBLK = 4608

def _codes(packed):
    """uint8 [..., K/2] -> uint8 codes [..., K] (element 2i = low nibble)."""
    lo = packed & 15
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


# Nibble slot i (the old layout's 4i..4i+3) -> bit positions of (m0, m1, m2, sign) in the 32-bit word
# (see convert8 in _CUDA_SRC).
_CODE_POS = {3: (6, 7, 8, 15), 7: (22, 23, 24, 31), 2: (2, 3, 4, 11), 6: (18, 19, 20, 27),
             1: (28, 29, 30, 5), 5: (12, 13, 14, 21), 0: (0, 1, 26, 9), 4: (16, 17, 10, 25)}  # dequant layout


def _repack(w, sf, gated):
    """w: uint8 [E, N, 2048]; sf: e4m3 [E, N, 256] -> uint8 [E, N/128, KS, WBLK]."""
    E, N, _ = w.shape
    n_tiles = N // 128
    dev = w.device
    if gated:
        nt = torch.arange(n_tiles, device=dev).view(n_tiles, 1, 1)
        q = torch.arange(8, device=dev).view(1, 8, 1)
        lr = torch.arange(16, device=dev).view(1, 1, 16)
        perm = torch.where(lr < 8, 64 * nt + 8 * q + lr, 4096 + 64 * nt + 8 * q + lr - 8).reshape(-1)
    else:
        perm = torch.arange(N, device=dev)
    # Keep persistent output out of cached segments left by repack temporaries.
    torch.cuda.empty_cache()
    out = torch.empty((E, n_tiles, KS, WBLK), dtype=torch.uint8, device=dev)
    sf_u8 = sf.view(torch.uint8)
    shifts = torch.tensor([3, 7, 1, 5, 2, 6, 0, 4], dtype=torch.int32, device=dev)  # (half, hk, b) order below
    pos = torch.tensor([_CODE_POS[int(i)] for i in [3, 7, 1, 5, 2, 6, 0, 4]], dtype=torch.int64, device=dev)  # [8, 4]
    for e in range(E):
        c = _codes(w[e][perm]).to(torch.int32)  # [N, 4096]
        # rows: [nt, wg, w, half, g]; k: [ks, j, hk, t, b]
        c = c.view(n_tiles, 2, 4, 2, 8, KS, 4, 2, 4, 2)
        # -> [nt, ks, wg, w, g, t, j, half, hk, b]
        c = c.permute(0, 5, 1, 2, 4, 8, 6, 3, 7, 9).reshape(n_tiles, KS, 2, 4, 8, 4, 4, 8)
        # Nibble index for (half, hk, b): i = 3 - (half + 2*hk) + 4*b; flattened order (half, hk, b):
        # (0,0,0)->3 (0,0,1)->7 (0,1,0)->1 (0,1,1)->5 (1,0,0)->2 (1,0,1)->6 (1,1,0)->0 (1,1,1)->4
        # Bit layout read by convert8 (rotate + mask per register): code bit b of the code in nibble slot i goes to
        # bit _CODE_POS[i][b] (b: m0, m1, m2, sign).
        cb = (c.to(torch.int64).unsqueeze(-1) >> torch.arange(4, device=dev)) & 1  # [..., 8, 4]
        del c
        words = (cb << pos).sum((-1, -2)) & 0xFFFFFFFF
        del cb
        wbytes = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], dim=-1).to(torch.uint8)
        out[e, :, :, :4096] = wbytes.reshape(n_tiles, KS, 4096)
        del words, wbytes
        s = sf_u8[e][perm].view(n_tiles, 2, 4, 2, 8, KS, 4)  # rows [nt, wg, w, half, g]; cols [ks, j]
        s = s.permute(0, 5, 1, 2, 4, 6, 3).reshape(n_tiles, KS, 512)  # [nt, ks, wg, w, g, j, half]
        out[e, :, :, 4096:] = s
        del s
    return out
