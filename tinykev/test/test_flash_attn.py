"""Packed Metal UIR Flash Attention (fused QK-norm+RoPE + block-causal attention + output gate) parity tests."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.llm.model import apply_rope, precompute_freqs_cis
from tinykev.flash_attn import get_packed_flash_attn, get_packed_flash_attn_wmma, get_qk_norm_rope
from tinykev.gather import gather_rows
from tinykev.packed import layout

class TestFlashAttn(unittest.TestCase):
  def test_packed_flash_attn_matches_row_sdpa(self):
    # Kev-0.8B exact attention dims: nh=8, nkv=2, hd=256, rope_dim=64, gated=True
    nh, nkv, hd, rope_dim = 8, 2, 256, 64
    S, blens = 22, [42, 15, 22]
    ids, seg, decide, opt_idx = list(range(S)), [0] * S, [], []
    for k_i, bl in enumerate(blens, start=1):
      base = len(ids)
      ids += list(range(base, base + bl)); seg += [k_i] * bl
      decide.append(base + bl - 1); opt_idx.append([base + 1])
    enc = {"ids": ids, "seg": seg, "pos": [], "decide_idx": decide, "opt_idx": opt_idx}
    lay = layout(enc)
    N, R, L = lay["N"], lay["R"], lay["L"]
    n_real = len(ids)

    tok_row, tok_pos = lay["tok_row"], lay["tok_pos"]

    k_off, v_off, P = nh * 2 * hd, nh * 2 * hd + nkv * hd, nh * 2 * hd + 2 * nkv * hd
    p = (Tensor.randn(N, P, dtype=dtypes.half) * 0.5).realize()
    qw = (Tensor.randn(hd, dtype=dtypes.half) * 0.2 + 1.0).realize()
    kw = (Tensor.randn(hd, dtype=dtypes.half) * 0.2 + 1.0).realize()
    freqs = precompute_freqs_cis(rope_dim, L + 16, 10000.0, device=p.device).realize()
    rows_t = Tensor(lay["rows_idx"], dtype=dtypes.int32, device=p.device).realize()
    pack_t = Tensor(lay["pack_idx"], dtype=dtypes.int32, device=p.device).realize()
    trow_t = Tensor(tok_row, dtype=dtypes.int32, device=p.device).realize()
    tpos_t = Tensor(tok_pos, dtype=dtypes.int32, device=p.device).realize()

    # Reference: exact row-form SDPA from packed.py
    q_raw, k_raw, v_raw = p[:, :k_off], p[:, k_off:v_off], p[:, v_off:]
    qg = q_raw.reshape(N, nh, 2, hd)
    q, gate = qg[:, :, 0, :], qg[:, :, 1, :].reshape(N, nh * hd)
    k, v = k_raw.reshape(N, nkv, hd), v_raw.reshape(N, nkv, hd)
    q = q.float().rmsnorm(eps=1e-6).cast(dtypes.half) * qw
    k = k.float().rmsnorm(eps=1e-6).cast(dtypes.half) * kw
    q, k, v = (gather_rows(z.reshape(N, -1), rows_t).reshape(R, L, -1, hd).transpose(1, 2) for z in (q, k, v))
    fr = freqs[0:L]
    q = apply_rope(q[..., :rope_dim], fr).cat(q[..., rope_dim:], dim=-1)
    k = apply_rope(k[..., :rope_dim], fr).cat(k[..., rope_dim:], dim=-1)
    attn = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True)
    ref = (gather_rows(attn.transpose(1, 2).cast(dtypes.half).reshape(R * L, -1), pack_t) * gate.sigmoid()).realize()

    # UIR Flash Attention (both scalar fallback and WMMA path)
    from tinygrad import UOp
    k1 = get_qk_norm_rope(N, nh, nkv, hd, rope_dim, P, k_off, True, True, 1e-6, freqs.shape[0])
    k2 = get_packed_flash_attn(N, R, L, nh, nkv, hd, P, v_off, True)
    qb = UOp.alloc((N, nh, hd), dtypes.half, device=p.device)
    kb = UOp.alloc((N, nkv, hd), dtypes.half, device=p.device)
    c1 = k1.call(qb, kb, p.uop, qw.uop, kw.uop, freqs.uop, tpos_t.uop)
    ob = UOp.alloc((N, nh * hd), dtypes.half, device=p.device)
    c2 = k2.call(ob, qb.after(c1), kb.after(c1), p.uop, rows_t.uop, trow_t.uop, tpos_t.uop)
    got = Tensor(ob.after(c2)).realize()

    diff = (got[:n_real].float() - ref[:n_real].float()).abs()
    self.assertLess(diff.max().item(), 1e-2)
    self.assertLess(diff.mean().item(), 1e-3)

    qt_t = Tensor(lay["q_tiles"], dtype=dtypes.int32, device=p.device).realize()
    k2w = get_packed_flash_attn_wmma(N, R, L, lay["NT"], nh, nkv, hd, P, v_off, True)
    obw = UOp.alloc((N, nh * hd), dtypes.half, device=p.device)
    c2w = k2w.call(obw, qb.after(c1), kb.after(c1), p.uop, rows_t.uop, qt_t.uop)
    got_w = Tensor(obw.after(c2w)).realize()
    diff_w = (got_w[:n_real].float() - ref[:n_real].float()).abs()
    self.assertLess(diff_w.max().item(), 1e-2)
    self.assertLess(diff_w.mean().item(), 1e-3)

if __name__ == "__main__": unittest.main()
