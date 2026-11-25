"""Chunked WY Gated DeltaNet (fused conv+norm+WY prep + WMMA chunk scan) parity tests."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, UOp, dtypes
from tinykev.chunked_delta import CHUNK_SIZE, get_chunk_scan, get_fused_conv_wy, make_chunks
from tinykev.conv import short_conv_packed
from tinykev.gated_delta import get_gated_delta_packed, packed_kernel_params

class TestChunkedDelta(unittest.TestCase):
  def test_make_chunks_multi_chunk_prefix_and_branches(self):
    # 40-token prefix (2 chunks of 32) + 3 branches of lengths 50 (2 chunks), 10 (1 chunk), 32 (1 chunk)
    toks, flags = make_chunks(40, [list(range(40, 90)), list(range(90, 100)), list(range(100, 132))], 144, 32)
    self.assertEqual(flags, [0, 1, 2, 0, 2, 2])
    self.assertEqual(len(toks), 6)
    self.assertEqual(toks[1][:8], list(range(32, 40)))
    self.assertTrue(all(x == 144 for x in toks[1][8:]))

  def test_fused_conv_wy_scan_matches_scalar(self):
    H, N, K, V, P = 4, 112, 128, 128, 4 * 128 * 4 + 2 * 4
    S, blens = 22, [42, 15, 22]
    branches, cur = [], S
    for bl in blens:
      branches.append(list(range(cur, cur + bl))); cur += bl
    toks, cflags = make_chunks(S, branches, N, CHUNK_SIZE)
    NC = len(toks)
    conv_src = [[N, N, N] for _ in range(N)]
    for b in branches:
      row = list(range(S)) + b
      for t, pidx in enumerate(row):
        for j in range(3):
          if t >= 3 - j: conv_src[pidx][j] = row[t - 3 + j]
    flags_p = [0] * N
    flags_p[S - 1] = 1
    for b in branches: flags_p[b[0]] = 2

    q_dim = H * K
    qa, qb, qc = 3 * q_dim, 4 * q_dim, 4 * q_dim + H
    p = (Tensor.randn(N, P, dtype=dtypes.half) * 0.4).realize()
    w_conv = (Tensor.randn(qa, 4, dtype=dtypes.half) * 0.4).realize()
    dt_bias = Tensor.randn(H, dtype=dtypes.float32).realize()
    ssm_a = (-Tensor.rand(H, dtype=dtypes.float32) * 0.5).realize()
    ctok_t = Tensor(toks, dtype=dtypes.int32).realize()
    cflags_t = Tensor(cflags, dtype=dtypes.int32).realize()
    csrc_t = Tensor(conv_src, dtype=dtypes.int32).realize()
    flags_t = Tensor(flags_p, dtype=dtypes.int32).realize()

    # Scalar reference pipeline
    beta = p[:, qb:qc].float().sigmoid().reshape(N, H)
    log_alpha = (p[:, qc:].float() + dt_bias).reshape(N, H).softplus() * ssm_a.reshape(H)
    conv = short_conv_packed(p[:, :qa].contiguous(), w_conv, csrc_t)
    q, k, v = conv.split([q_dim, q_dim, q_dim], dim=-1)
    q, k = ((z.reshape(N, H, K).rmsnorm(eps=1e-6 / K) / K**0.5) for z in (q, k))
    v = v.reshape(N, H, V)
    q, k, v = (z.transpose(0, 1).float() for z in (q, k, v))
    q = q * K**-0.5
    kq = (q * k).sum(-1).contiguous()
    kern_ref = get_gated_delta_packed().specialize(**packed_kernel_params(heads=H, tokens=N, k_blks=K // 32, v_tiles=V // 4, alpha_k=1))
    out_ref = UOp.alloc((H, N, V), dtypes.float32, device=p.device)
    ref = Tensor(out_ref.after(kern_ref.call(out_ref, q.contiguous().uop, k.contiguous().uop, v.contiguous().uop,
                                             beta.transpose(0, 1).contiguous().uop,
                                             log_alpha.transpose(0, 1).exp().reshape(H, N, 1).contiguous().uop,
                                             kq.uop, flags_t.uop))).realize()

    # Fused chunked WY pipeline
    k_prep = get_fused_conv_wy(H, N, NC, P, K, V, 0, q_dim, 2 * q_dim, qb, qc)
    k_scan = get_chunk_scan(H, N, NC, K, V)
    dev = p.device
    W, U = UOp.alloc((H, NC, 32, K), dtypes.half, device=dev), UOp.alloc((H, NC, 32, V), dtypes.half, device=dev)
    Qg, Kg = UOp.alloc((H, NC, 32, K), dtypes.half, device=dev), UOp.alloc((H, NC, 32, K), dtypes.half, device=dev)
    L, gl = UOp.alloc((H, NC, 32, 32), dtypes.half, device=dev), UOp.alloc((H, NC), dtypes.float32, device=dev)
    c1 = k_prep.call(W, U, Qg, Kg, L, gl, p.uop, w_conv.uop, dt_bias.uop, ssm_a.uop, ctok_t.uop, csrc_t.uop)
    out_got = UOp.alloc((H, N, V), dtypes.float32, device=dev)
    c2 = k_scan.call(out_got, W.after(c1), U.after(c1), Qg.after(c1), Kg.after(c1), L.after(c1), gl.after(c1), ctok_t.uop, cflags_t.uop)
    got = Tensor(out_got.after(c2)).realize()

    n_real = S + sum(blens)
    diff = (got[:, :n_real] - ref[:, :n_real]).abs()
    self.assertLess(diff.max().item(), 2e-3)
    self.assertLess(diff.mean().item(), 2e-4)

    # Cached-prefix split: run prefix alone (save state + last-3 conv inputs), then branches with prefix
    N_p, N_b = 32, 80
    toks_p, flags_p_ch = make_chunks(S, [], N_p, CHUNK_SIZE)
    csrc_p = [[N_p, N_p, N_p] for _ in range(N_p)]
    for t in range(S):
      for j in range(3):
        if t >= 3 - j: csrc_p[t][j] = t - 3 + j
    p_pref_in = p[:S].cat(Tensor.zeros(N_p - S, P, dtype=dtypes.half, device=dev), dim=0).realize()
    kp_prep = get_fused_conv_wy(H, N_p, len(toks_p), P, K, V, 0, q_dim, 2 * q_dim, qb, qc)
    kp_scan = get_chunk_scan(H, N_p, len(toks_p), K, V, has_init=False, save_out=True)
    Wp, Up = UOp.alloc((H, len(toks_p), 32, K), dtypes.half, device=dev), UOp.alloc((H, len(toks_p), 32, V), dtypes.half, device=dev)
    Qgp, Kgp = UOp.alloc((H, len(toks_p), 32, K), dtypes.half, device=dev), UOp.alloc((H, len(toks_p), 32, K), dtypes.half, device=dev)
    Lp, glp = UOp.alloc((H, len(toks_p), 32, 32), dtypes.half, device=dev), UOp.alloc((H, len(toks_p)), dtypes.float32, device=dev)
    ct_p, cf_p, cs_p = (Tensor(x, dtype=dtypes.int32).realize() for x in (toks_p, flags_p_ch, csrc_p))
    cp1 = kp_prep.call(Wp, Up, Qgp, Kgp, Lp, glp, p_pref_in.uop, w_conv.uop, dt_bias.uop, ssm_a.uop, ct_p.uop, cs_p.uop)
    out_p, s_out = UOp.alloc((H, N_p, V), dtypes.float32, device=dev), UOp.alloc((H, V, K), dtypes.half, device=dev)
    cp2 = kp_scan.call(out_p, Wp.after(cp1), Up.after(cp1), Qgp.after(cp1), Kgp.after(cp1), Lp.after(cp1), glp.after(cp1), ct_p.uop, cf_p.uop, s_out)
    saved_S = Tensor(s_out.after(cp2)).realize()
    conv_st = p[S - 3 : S, :qa].contiguous().realize()

    b_shifted = [[idx - S for idx in b] for b in branches]
    toks_b, flags_b_ch = make_chunks(0, b_shifted, N_b, CHUNK_SIZE)
    csrc_b = [[N_b + 3, N_b + 3, N_b + 3] for _ in range(N_b)]
    for b in b_shifted:
      for tb, pb in enumerate(b):
        for j in range(3):
          off = tb - 3 + j
          csrc_b[pb][j] = b[off] if off >= 0 else N_b + 3 + off
    n_b = sum(blens)
    p_b = p[S : S + n_b].cat(Tensor.zeros(N_b - n_b, P, dtype=dtypes.half, device=dev), dim=0).realize()
    kb_prep = get_fused_conv_wy(H, N_b, len(toks_b), P, K, V, 0, q_dim, 2 * q_dim, qb, qc, has_prefix=True)
    kb_scan = get_chunk_scan(H, N_b, len(toks_b), K, V, has_init=True, save_out=False)
    Wb, Ub = UOp.alloc((H, len(toks_b), 32, K), dtypes.half, device=dev), UOp.alloc((H, len(toks_b), 32, V), dtypes.half, device=dev)
    Qgb, Kgb = UOp.alloc((H, len(toks_b), 32, K), dtypes.half, device=dev), UOp.alloc((H, len(toks_b), 32, K), dtypes.half, device=dev)
    Lb, glb = UOp.alloc((H, len(toks_b), 32, 32), dtypes.half, device=dev), UOp.alloc((H, len(toks_b)), dtypes.float32, device=dev)
    ct_b, cf_b, cs_b = (Tensor(x, dtype=dtypes.int32).realize() for x in (toks_b, flags_b_ch, csrc_b))
    cb1 = kb_prep.call(Wb, Ub, Qgb, Kgb, Lb, glb, p_b.uop, w_conv.uop, dt_bias.uop, ssm_a.uop, ct_b.uop, cs_b.uop, conv_st.uop)
    out_b = UOp.alloc((H, N_b, V), dtypes.float32, device=dev)
    cb2 = kb_scan.call(out_b, Wb.after(cb1), Ub.after(cb1), Qgb.after(cb1), Kgb.after(cb1), Lb.after(cb1), glb.after(cb1), ct_b.uop, cf_b.uop, saved_S.uop)
    got_b = Tensor(out_b.after(cb2)).realize()
    diff_b = (got_b[:, :n_b] - got[:, S : S + n_b]).abs()
    self.assertLess(diff_b.max().item(), 2e-3)
    self.assertLess(diff_b.mean().item(), 2e-4)

if __name__ == "__main__": unittest.main()
