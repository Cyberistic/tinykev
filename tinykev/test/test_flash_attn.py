"""Packed Metal UIR Flash Attention (fused QK-norm+RoPE + block-causal attention + output gate) parity tests."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.llm.model import apply_rope, precompute_freqs_cis
from tinykev.flash_attn import get_packed_flash_attn, get_packed_flash_attn_wmma, get_qk_norm_rope, get_ro_flash_attn, get_ro_qk_norm_rope
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

    # Readout-only path: matches full packed Flash Attention at exact readout indices (decide + opts)
    M_ro = lay["M_ro"]
    ro_idx = lay["ro_idx"]
    ro_row_t = Tensor(lay["ro_row"], dtype=dtypes.int32, device=p.device).realize()
    ro_pos_t = Tensor(lay["ro_pos"], dtype=dtypes.int32, device=p.device).realize()
    ro_idx_t = Tensor(ro_idx, dtype=dtypes.int32, device=p.device).realize()
    q_ro_p = gather_rows(p[:, :k_off].contiguous(), ro_idx_t).realize()
    kv_all_p = p[:, k_off:].contiguous().realize()
    kr1 = get_ro_qk_norm_rope(M_ro, N, nh, nkv, hd, rope_dim, True, True, 1e-6, freqs.shape[0])
    kr2 = get_ro_flash_attn(M_ro, N, R, L, nh, nkv, hd, True)
    qrb = UOp.alloc((M_ro, nh, hd), dtypes.half, device=p.device)
    krb = UOp.alloc((N, nkv, hd), dtypes.half, device=p.device)
    cr1 = kr1.call(qrb, krb, q_ro_p.uop, kv_all_p.uop, qw.uop, kw.uop, freqs.uop, ro_pos_t.uop, tpos_t.uop)
    orb = UOp.alloc((M_ro, nh * hd), dtypes.half, device=p.device)
    cr2 = kr2.call(orb, qrb.after(cr1), krb.after(cr1), kv_all_p.uop, q_ro_p.uop, rows_t.uop, ro_row_t.uop, ro_pos_t.uop)
    got_ro = Tensor(orb.after(cr2)).realize()
    n_ro = len(lay["decide"]) + len(lay["opts"])
    ref_ro = gather_rows(ref, ro_idx_t).realize()
    diff_ro = (got_ro[:n_ro].float() - ref_ro[:n_ro].float()).abs()
    self.assertLess(diff_ro.max().item(), 1e-2)
    self.assertLess(diff_ro.mean().item(), 1e-3)

    # Zero-copy cached-prefix WMMA + readout-only Flash Attention must match full packed attention
    from tinykev.packed import layout_branches
    lay_b = layout_branches(enc, S)
    N_b, n_b = lay_b["N"], n_real - S
    p_b = p[S:n_real].cat(Tensor.zeros(N_b - n_b, P, dtype=dtypes.half, device=p.device), dim=0).realize()
    k_pref = Tensor(kb.after(c1))[:S].contiguous().realize()
    v_pref = p[:S, v_off:].contiguous().realize()
    rows_b_t = Tensor(lay_b["rows_idx"], dtype=dtypes.int32, device=p.device).realize()
    tpos_b_t = Tensor(lay_b["tok_pos"], dtype=dtypes.int32, device=p.device).realize()
    qt_b_t = Tensor(lay_b["q_tiles"], dtype=dtypes.int32, device=p.device).realize()
    kb1 = get_qk_norm_rope(N_b, nh, nkv, hd, rope_dim, P, k_off, True, True, 1e-6, freqs.shape[0])
    qbb, kbb = UOp.alloc((N_b, nh, hd), dtypes.half, device=p.device), UOp.alloc((N_b, nkv, hd), dtypes.half, device=p.device)
    cb1 = kb1.call(qbb, kbb, p_b.uop, qw.uop, kw.uop, freqs.uop, tpos_b_t.uop)
    kb2 = get_packed_flash_attn_wmma(N_b, lay_b["R"], lay_b["L"], lay_b["NT"], nh, nkv, hd, P, v_off, True, S)
    obb = UOp.alloc((N_b, nh * hd), dtypes.half, device=p.device)
    cb2 = kb2.call(obb, qbb.after(cb1), kbb.after(cb1), p_b.uop, rows_b_t.uop, qt_b_t.uop, k_pref.uop, v_pref.uop)
    got_bp = Tensor(obb.after(cb2)).realize()
    diff_bp = (got_bp[:n_b].float() - got_w[S:n_real].float()).abs()
    self.assertLess(diff_bp.max().item(), 1e-2)
    self.assertLess(diff_bp.mean().item(), 1e-3)

    ro_b_idx_t = Tensor(lay_b["ro_idx"], dtype=dtypes.int32, device=p.device).realize()
    ro_b_row_t = Tensor(lay_b["ro_row"], dtype=dtypes.int32, device=p.device).realize()
    ro_b_pos_t = Tensor(lay_b["ro_pos"], dtype=dtypes.int32, device=p.device).realize()
    q_rob_p = gather_rows(p_b[:, :k_off].contiguous(), ro_b_idx_t).realize()
    kv_b_p = p_b[:, k_off:].contiguous().realize()
    krb1 = get_ro_qk_norm_rope(lay_b["M_ro"], N_b, nh, nkv, hd, rope_dim, True, True, 1e-6, freqs.shape[0])
    krb2 = get_ro_flash_attn(lay_b["M_ro"], N_b, lay_b["R"], lay_b["L"], nh, nkv, hd, True, S)
    qrbb = UOp.alloc((lay_b["M_ro"], nh, hd), dtypes.half, device=p.device)
    krbb = UOp.alloc((N_b, nkv, hd), dtypes.half, device=p.device)
    crb1 = krb1.call(qrbb, krbb, q_rob_p.uop, kv_b_p.uop, qw.uop, kw.uop, freqs.uop, ro_b_pos_t.uop, tpos_b_t.uop)
    orbb = UOp.alloc((lay_b["M_ro"], nh * hd), dtypes.half, device=p.device)
    crb2 = krb2.call(orbb, qrbb.after(crb1), krbb.after(crb1), kv_b_p.uop, q_rob_p.uop, rows_b_t.uop, ro_b_row_t.uop, ro_b_pos_t.uop,
                     k_pref.uop, v_pref.uop)
    got_rob = Tensor(orbb.after(crb2)).realize()
    diff_rob = (got_rob[:n_ro].float() - got_ro[:n_ro].float()).abs()
    self.assertLess(diff_rob.max().item(), 1e-2)
    self.assertLess(diff_rob.mean().item(), 1e-3)

if __name__ == "__main__": unittest.main()
