"""Chunked WY Gated DeltaNet for Metal: fused conv+norm+WY prep + WMMA chunk recurrence.

Replaces the 10-kernel pre-scan DeltaNet pipeline (slice, beta, alpha, short_conv, q/k RMSNorm, transposes, kq)
and the 112-step scalar `simd_sum` recurrent scan with two `< 14 KB`-LDS Metal UIR kernels:

1. `fused_conv_wy_c32`: reads the fused projection `p [N, P]` directly (no slice) for each `(head, 32-token chunk)`,
   runs the 4-tap packed causal conv + SiLU + warp `simd_sum` L2 norms (zero barriers inside the conv loop), computes
   `beta`, `log_alpha`, cumulative log-decay `g` via `simd_prefix_inclusive_sum`, computes `KK^T`/`QK^T` via `UOp.wmma`,
   inverts `A = (I + M)^{-1}` via parallel 2x16 block inversion, and emits `W, U, Qg, Kg, L, g_last`.
2. `chunk_scan_c32`: walks the `NC` chunks (only 5 chunks for the 101-token request vs 112 serial steps) with `S`
   resident in registers (16 floats/thread) and `UOp.wmma` for `WS^T`, `Qg S^T`, `L U_new`, and `U_new^T Kg`,
   saving/restoring the shared prefix state in threadgroup memory at chunk flags `1` / `2`.
"""
from __future__ import annotations

import functools, math, os, pathlib, threading

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
CHUNK_SIZE = 32
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)

def enabled() -> bool: return os.environ.get("TINYKEV_CHUNKED_DELTA", "1") != "0"

def _is_metal(device) -> bool:
  d = device if isinstance(device, str) else (device[0] if isinstance(device, tuple) and device else None)
  return isinstance(d, str) and d.split(":")[0] == "METAL"

def _silu(x:UOp) -> UOp: return x * ((-x * LOG2E).exp2() + 1.0).reciprocal()
def _sigmoid(x:UOp) -> UOp: return ((-x * LOG2E).exp2() + 1.0).reciprocal()
def _softplus(x:UOp) -> UOp: return x.maximum(0.0) + ((-x.abs() * LOG2E).exp2() + 1.0).log2() * LN2

def make_chunks(S:int, branches:list[list[int]], N:int, C:int=CHUNK_SIZE) -> tuple[list[list[int]], list[int]]:
  """Partition the state (`0..S-1`) and each branch into `C`-token chunks (`N` marks pad slots).

  Flags: `1` saves the recurrent state after the last prefix chunk, `2` restores it before each branch's first chunk."""
  toks, flags = [], []
  def add_seq(indices:list[int], is_state:bool):
    n_ch = max(1, (len(indices) + C - 1) // C)
    for ci in range(n_ch):
      sl = indices[ci * C : (ci + 1) * C]
      toks.append(sl + [N] * (C - len(sl)))
      flags.append((1 if ci == n_ch - 1 else 0) if is_state else (2 if ci == 0 else 0))
  if S > 0: add_seq(list(range(S)), is_state=True)
  for b in branches: add_seq(b, is_state=False)
  return toks, flags

def _emit(key:str, u:UOp) -> UOp:
  with Context(SPEC=0):
    uir = render_uir(graph_rewrite(u, symbolic_simple, enter_calls=True), color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"{key}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def _load2(buf_flat:UOp, off:UOp) -> UOp:
  return UOp(Ops.SHRINK, src=(buf_flat, off, UOp.const(2))).load()

def _load4(buf_flat:UOp, off:UOp) -> UOp:
  return UOp(Ops.SHRINK, src=(buf_flat, off, UOp.const(4))).load()

def _build_fused_conv_wy(H:int, N:int, NC:int, P:int, K:int, V:int,
                         q_off:int, k_off:int, v_off:int, beta_off:int, alpha_off:int, has_prefix:bool=False) -> UOp:
  C = CHUNK_SIZE
  assert K == 128 and V == 128 and C == 32
  inv_sqrt_k = K ** -0.5

  W_out = UOp.param(0, dtypes.half, (H, NC, C, K))
  U_out = UOp.param(1, dtypes.half, (H, NC, C, V))
  Qg_out = UOp.param(2, dtypes.half, (H, NC, C, K))
  Kg_out = UOp.param(3, dtypes.half, (H, NC, C, K))
  L_out = UOp.param(4, dtypes.half, (H, NC, C, C))
  gl_out = UOp.param(5, dtypes.float32, (H, NC))
  p_in = UOp.param(6, dtypes.half, (N, P))
  w_conv = UOp.param(7, dtypes.half, (v_off + H * V, 4))
  dt_bias = UOp.param(8, dtypes.float32, (H,))
  ssm_a = UOp.param(9, dtypes.float32, (H,))
  ctok = UOp.param(10, dtypes.int32, (NC, C))
  csrc = UOp.param(11, dtypes.int32, (N, 3))
  p_pref = UOp.param(12, dtypes.half, (3, v_off + H * V)) if has_prefix else None

  hc = UOp.range(H * NC, 0, AxisType.GLOBAL)
  h, c = hc // NC, hc % NC
  wid = UOp.range(4, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  wm, wn = wid // 2, wid % 2
  tid = wid * 32 + lane

  m_lane = ((lane >> 1) & 3) + ((lane >> 4) * 4)
  n_base = ((lane & 1) * 2) + (((lane >> 3) & 1) * 4)

  # <= 10.62 KB LDS total -> 3 threadgroups/core (57 TGs/wave = 2 waves instead of 3 on 19-core M2 Pro!)
  # Single MA_raw buffer holds M, then in-place diagonal A00/A11, then T in upper-right and A10 in lower-left.
  K_STR, C_STR = K, C + 2
  K_raw = UOp.alloc((C * K_STR,), dtypes.half, addrspace=AddrSpace.LOCAL)
  MA_raw = UOp.alloc((C * C_STR,), dtypes.half, addrspace=AddrSpace.LOCAL)
  K_lds, MA_lds = K_raw.reshape(C, K_STR), MA_raw.reshape(C, C_STR)
  g_lds = UOp.alloc((C,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  b_lds = UOp.alloc((C,), dtypes.float32, addrspace=AddrSpace.LOCAL)

  # 1. beta, log_alpha, and cumulative log-decay g[0..31] in warp 0
  tok_l = ctok[c, lane].load()
  valid_l = (tok_l >= 0) & (tok_l < N)
  tok_s = tok_l.maximum(0).minimum(N - 1)
  raw_b = p_in[tok_s, beta_off + h].load().float()
  raw_a = p_in[tok_s, alpha_off + h].load().float() + dt_bias[h].load()
  bt_v = valid_l.where(_sigmoid(raw_b), UOp.const(0.0, dtypes.float32))
  la_v = valid_l.where(_softplus(raw_a) * ssm_a[h].load(), UOp.const(0.0, dtypes.float32))
  g_v = UOp.native("simd_prefix_inclusive_sum({0})", shape=(), dtype=dtypes.float32, fmt=True).call(la_v)
  st_gb = UOp.group(
    g_lds[lane.valid(wid.eq(0))].store(g_v),
    b_lds[lane.valid(wid.eq(0))].store(bt_v),
    gl_out[h, c.valid(tid.eq(31))].store((g_v * LOG2E).exp2()),
  ).barrier()
  g_r, b_r = g_lds.after(st_gb), b_lds.after(st_gb)
  g_last_v = g_r[31].load()

  # Preload w_conv as float32 for this lane's 4 channels (Q, K, V) x 4 taps before the 8-step loop
  wq = [[w_conv[q_off + h * K + lane * 4 + e, j].load().float() for j in range(4)] for e in range(4)]
  wk = [[w_conv[k_off + h * K + lane * 4 + e, j].load().float() for j in range(4)] for e in range(4)]
  wv = [[w_conv[v_off + h * V + lane * 4 + e, j].load().float() for j in range(4)] for e in range(4)]

  # 2. 8-step zero-barrier loop: each warp handles 8 tokens, each lane owns 4 contiguous channels (`half4` loads) -> simd_sum L2 norm!
  p_flat = p_in.flatten()
  pp_flat = p_pref.flatten() if has_prefix else None
  qa_stride = v_off + H * V
  q_base = q_off + h * K + lane * 4
  k_base_c = k_off + h * K + lane * 4
  v_base_c = v_off + h * V + lane * 4

  step = UOp.range(8, 10)
  t_cur = step * 4 + wid
  tok_r = ctok[c, t_cur].load()
  vr = (tok_r >= 0) & (tok_r < N)
  tok_rs = tok_r.maximum(0).minimum(N - 1)
  s_idx = [csrc[tok_rs, j].load() for j in range(3)]
  s_val = [(idx >= 0) & (idx < N) for idx in s_idx]
  s_safe = [idx.maximum(0).minimum(N - 1) for idx in s_idx]
  if has_prefix:
    s_pref = [(idx >= N) & (idx < N + 3) for idx in s_idx]
    s_pi = [(idx - N).maximum(0).minimum(2) for idx in s_idx]
    q3 = _load4(p_flat, tok_rs * P + q_base)
    k3 = _load4(p_flat, tok_rs * P + k_base_c)
    v3 = _load4(p_flat, tok_rs * P + v_base_c)
    qa_l = [q3[e].float() * wq[e][3] for e in range(4)]
    ka_l = [k3[e].float() * wk[e][3] for e in range(4)]
    va_l = [v3[e].float() * wv[e][3] for e in range(4)]
    for j in range(3):
      q_t = _load4(p_flat, s_safe[j] * P + q_base)
      k_t = _load4(p_flat, s_safe[j] * P + k_base_c)
      v_t = _load4(p_flat, s_safe[j] * P + v_base_c)
      qp_t = _load4(pp_flat, (s_pi[j] * qa_stride + q_base).valid(s_pref[j]))
      kp_t = _load4(pp_flat, (s_pi[j] * qa_stride + k_base_c).valid(s_pref[j]))
      vp_t = _load4(pp_flat, (s_pi[j] * qa_stride + v_base_c).valid(s_pref[j]))
      for e in range(4):
        qa_l[e] = qa_l[e] + s_val[j].where(q_t[e].float(), qp_t[e].float()) * wq[e][j]
        ka_l[e] = ka_l[e] + s_val[j].where(k_t[e].float(), kp_t[e].float()) * wk[e][j]
        va_l[e] = va_l[e] + s_val[j].where(v_t[e].float(), vp_t[e].float()) * wv[e][j]
  else:
    q3 = _load4(p_flat, (tok_rs * P + q_base).valid(vr))
    k3 = _load4(p_flat, (tok_rs * P + k_base_c).valid(vr))
    v3 = _load4(p_flat, (tok_rs * P + v_base_c).valid(vr))
    qa_l = [q3[e].float() * wq[e][3] for e in range(4)]
    ka_l = [k3[e].float() * wk[e][3] for e in range(4)]
    va_l = [v3[e].float() * wv[e][3] for e in range(4)]
    for j in range(3):
      q_t = _load4(p_flat, (s_safe[j] * P + q_base).valid(vr & s_val[j]))
      k_t = _load4(p_flat, (s_safe[j] * P + k_base_c).valid(vr & s_val[j]))
      v_t = _load4(p_flat, (s_safe[j] * P + v_base_c).valid(vr & s_val[j]))
      for e in range(4):
        qa_l[e] = qa_l[e] + q_t[e].float() * wq[e][j]
        ka_l[e] = ka_l[e] + k_t[e].float() * wk[e][j]
        va_l[e] = va_l[e] + v_t[e].float() * wv[e][j]

  q_vals, k_vals, v_vals = [], [], []
  q_sq_l, k_sq_l = UOp.const(0.0, dtypes.float32), UOp.const(0.0, dtypes.float32)
  for e in range(4):
    qv0, kv0, vv0 = vr.where(_silu(qa_l[e]), 0.0), vr.where(_silu(ka_l[e]), 0.0), vr.where(_silu(va_l[e]), 0.0)
    q_vals.append(qv0); k_vals.append(kv0); v_vals.append(vv0)
    q_sq_l = q_sq_l + qv0 * qv0
    k_sq_l = k_sq_l + kv0 * kv0

  q_inv = (UOp.native("simd_sum({0})", shape=(), dtype=dtypes.float32, fmt=True).call(q_sq_l) + 1e-6).rsqrt() * inv_sqrt_k
  k_inv = (UOp.native("simd_sum({0})", shape=(), dtype=dtypes.float32, fmt=True).call(k_sq_l) + 1e-6).rsqrt()
  gs, bs = g_r[t_cur].load(), b_r[t_cur].load()
  exp_g = (gs * LOG2E).exp2()
  exp_gl_g = ((g_last_v - gs).minimum(0.0) * LOG2E).exp2()

  st_c = []
  for e in range(4):
    ch = lane * 4 + e
    qn, kn = q_vals[e] * q_inv, k_vals[e] * k_inv
    st_c += [K_lds.after(st_gb)[t_cur, ch].store(kn.cast(dtypes.half)),
             W_out[h, c, t_cur, ch].store(qn.cast(dtypes.half)),                # temp unscaled q_norm
             U_out[h, c, t_cur, ch].store((v_vals[e] * bs).cast(dtypes.half)),  # temp Vs = beta * v
             Qg_out[h, c, t_cur, ch].store((qn * exp_g).cast(dtypes.half)),
             Kg_out[h, c, t_cur, ch].store((kn * exp_gl_g).cast(dtypes.half))]
  st_conv = UOp.group(*st_c).end(step).barrier()

  K_raw_r, K_r = K_raw.after(st_conv), K_lds.after(st_conv)
  Q_tmp_flat = W_out.after(st_conv).flatten()
  Vs_tmp = U_out.after(st_conv)

  # 3. Looped KK^T and QK^T via UOp.wmma (`half2` reads; kept in a K/8 loop to avoid register spill)
  kk_acc = UOp.alloc((2, 2, 2), dtypes.float32, addrspace=AddrSpace.REG)
  qk_acc = UOp.alloc((2, 2, 2), dtypes.float32, addrspace=AddrSpace.REG)
  kk_acc = kk_acc.after(*[kk_acc[i, j, e].store(0.0) for i in range(2) for j in range(2) for e in range(2)])
  qk_acc = qk_acc.after(*[qk_acc[i, j, e].store(0.0) for i in range(2) for j in range(2) for e in range(2)])

  ks = UOp.range(K // 8, 20)
  k0 = ks * 8
  ckk = [[UOp.stack(kk_acc.after(ks)[i, j, 0].load(), kk_acc.after(ks)[i, j, 1].load()) for j in range(2)] for i in range(2)]
  cqk = [[UOp.stack(qk_acc.after(ks)[i, j, 0].load(), qk_acc.after(ks)[i, j, 1].load()) for j in range(2)] for i in range(2)]
  kf_a = [_load2(K_raw_r, (wm * 16 + i * 8 + m_lane) * K_STR + k0 + n_base) for i in range(2)]
  qf_a = [_load2(Q_tmp_flat, ((h * NC + c) * C + wm * 16 + i * 8 + m_lane) * K + k0 + n_base) for i in range(2)]
  kf_b = [UOp.stack(K_r[wn * 16 + j * 8 + n_base, k0 + m_lane].load(),
                    K_r[wn * 16 + j * 8 + n_base + 1, k0 + m_lane].load()) for j in range(2)]
  for i in range(2):
    for j in range(2):
      ckk[i][j] = UOp.wmma(kf_a[i], kf_b[j], ckk[i][j], (8, 8, 8), 32)
      cqk[i][j] = UOp.wmma(qf_a[i], kf_b[j], cqk[i][j], (8, 8, 8), 32)
  upd_g = UOp.group(
    *[kk_acc[i, j, e].store(ckk[i][j][e]) for i in range(2) for j in range(2) for e in range(2)],
    *[qk_acc[i, j, e].store(cqk[i][j][e]) for i in range(2) for j in range(2) for e in range(2)],
  ).end(ks)
  kk_f, qk_f = kk_acc.after(upd_g), qk_acc.after(upd_g)

  st_ml = []
  for i in range(2):
    ri = wm * 16 + i * 8 + m_lane
    gi, bi = g_r[ri].load(), b_r[ri].load()
    for j in range(2):
      for e in range(2):
        cj = wn * 16 + j * 8 + n_base + e
        gj = g_r[cj].load()
        decay = ((gi - gj).minimum(0.0) * LOG2E).exp2()
        mv = (cj < ri).where(bi * decay * kk_f[i, j, e].load(), UOp.const(0.0, dtypes.float32))
        lv = (cj <= ri).where(decay * qk_f[i, j, e].load(), UOp.const(0.0, dtypes.float32))
        st_ml += [MA_lds[ri, cj].store(mv.cast(dtypes.half)), L_out[h, c, ri, cj].store(lv.cast(dtypes.half))]
  b_ml = UOp.group(*st_ml).barrier()
  MA_r = MA_lds.after(b_ml)

  # 4a. Parallel 2x16 block-diagonal inversion in warp 0 (computes all 16 rows in regs, then writes in-place to MA_lds)
  blk, j16 = lane // 16, lane % 16
  base16 = blk * 16
  a_col = [UOp.const(0.0, dtypes.float32) for _ in range(16)]
  for i in range(16):
    if i == 0:
      val = j16.eq(0).where(UOp.const(1.0, dtypes.float32), UOp.const(0.0, dtypes.float32))
    else:
      acc_s = UOp.const(0.0, dtypes.float32)
      for m in range(i): acc_s = acc_s + MA_r[base16 + i, base16 + m].load().float() * a_col[m]
      val = j16.eq(i).where(UOp.const(1.0, dtypes.float32), (j16 < i).where(-acc_s, UOp.const(0.0, dtypes.float32)))
    a_col[i] = val
  st_diag = [MA_lds.after(b_ml)[(base16 + i).valid(wid.eq(0)), base16 + j16].store(a_col[i].cast(dtypes.half)) for i in range(16)]

  si = UOp.range(C, 40)
  scale_k = b_r[si].load() * (g_r[si].load() * LOG2E).exp2()
  st_ks = K_lds.after(b_ml)[si, tid].store((K_r[si, tid].load().float() * scale_k).cast(dtypes.half)).end(si)
  b_diag = UOp.group(*st_diag, st_ks).barrier()
  MA_d_raw, Ks_raw = MA_raw.after(b_diag), K_raw.after(b_diag)

  # 4b. Off-diagonal block A10 = -A11 @ (M10 @ A00) via UOp.wmma (T stored in unused upper-right MA[0:16, 16:32])
  rm16, cn16 = wm * 8 + m_lane, wn * 8 + n_base
  t_acc = UOp.const((0.0, 0.0), dtypes.float32)
  for kk in range(2):
    k0 = kk * 8
    mf_a = _load2(MA_d_raw, (16 + rm16) * C_STR + k0 + n_base)
    af_b = _load2(MA_d_raw, (k0 + m_lane) * C_STR + cn16)
    t_acc = UOp.wmma(mf_a, af_b, t_acc, (8, 8, 8), 32)
  b_t = UOp.group(MA_lds.after(b_diag)[rm16, 16 + cn16].store(t_acc[0].cast(dtypes.half)),
                  MA_lds.after(b_diag)[rm16, 16 + cn16 + 1].store(t_acc[1].cast(dtypes.half))).barrier()
  MA_t_raw = MA_raw.after(b_t)

  a10 = UOp.const((0.0, 0.0), dtypes.float32)
  for kk in range(2):
    k0 = kk * 8
    a11_a = _load2(MA_t_raw, (16 + rm16) * C_STR + 16 + k0 + n_base)
    t_b = _load2(MA_t_raw, (k0 + m_lane) * C_STR + 16 + cn16)
    a10 = UOp.wmma(a11_a, t_b, a10, (8, 8, 8), 32)
  b_a = UOp.group(MA_lds.after(b_t)[16 + rm16, cn16].store((-a10[0]).cast(dtypes.half)),
                  MA_lds.after(b_t)[16 + rm16, cn16 + 1].store((-a10[1]).cast(dtypes.half))).barrier()
  A_raw_r = MA_raw.after(b_a)

  # 5. Looped W = A @ Ks -> W_out, reload Vs -> K_lds, looped U = A @ Vs -> U_out
  # Mask out upper-right quadrant (wm == 0, kk >= 2 -> 0) since MA[0:16, 16:32] still holds T.
  nb_w = UOp.range(2, 50)
  n_off_w = (wn * 2 + nb_w) * 32
  w_acc = UOp.alloc((2, 4, 2), dtypes.float32, addrspace=AddrSpace.REG)
  w_acc = w_acc.after(*[w_acc.after(nb_w)[i, j, e].store(0.0) for i in range(2) for j in range(4) for e in range(2)])
  kk_w = UOp.range(4, 51)
  kw0 = kk_w * 8
  in_tri_w = wm.eq(1) | (kk_w < 2)
  cw = [[UOp.stack(w_acc.after(kk_w)[i, j, 0].load(), w_acc.after(kk_w)[i, j, 1].load()) for j in range(4)] for i in range(2)]
  af_w = [in_tri_w.where(_load2(A_raw_r, (wm * 16 + i * 8 + m_lane) * C_STR + kw0 + n_base), UOp.const((0.0, 0.0), dtypes.half)) for i in range(2)]
  ksf = [_load2(Ks_raw, (kw0 + m_lane) * K_STR + n_off_w + j * 8 + n_base) for j in range(4)]
  for i in range(2):
    for j in range(4): cw[i][j] = UOp.wmma(af_w[i], ksf[j], cw[i][j], (8, 8, 8), 32)
  wf = w_acc.after(UOp.group(*[w_acc[i, j, e].store(cw[i][j][e]) for i in range(2) for j in range(4) for e in range(2)]).end(kk_w))
  st_w = [W_out.after(b_a)[h, c, wm * 16 + i * 8 + m_lane, n_off_w + j * 8 + n_base + e].store(wf[i, j, e].load().cast(dtypes.half))
          for i in range(2) for j in range(4) for e in range(2)]
  b_w = UOp.group(*st_w).end(nb_w).barrier()

  vi_l = UOp.range(C, 52)
  b_vs = K_lds.after(b_w)[vi_l, tid].store(Vs_tmp[h, c, vi_l, tid].load()).end(vi_l).barrier()
  Vs_raw = K_raw.after(b_vs)

  nb_u = UOp.range(2, 53)
  n_off_u = (wn * 2 + nb_u) * 32
  u_acc = UOp.alloc((2, 4, 2), dtypes.float32, addrspace=AddrSpace.REG)
  u_acc = u_acc.after(*[u_acc.after(nb_u)[i, j, e].store(0.0) for i in range(2) for j in range(4) for e in range(2)])
  kk_u = UOp.range(4, 54)
  ku0 = kk_u * 8
  in_tri_u = wm.eq(1) | (kk_u < 2)
  cu = [[UOp.stack(u_acc.after(kk_u)[i, j, 0].load(), u_acc.after(kk_u)[i, j, 1].load()) for j in range(4)] for i in range(2)]
  af_u = [in_tri_u.where(_load2(A_raw_r, (wm * 16 + i * 8 + m_lane) * C_STR + ku0 + n_base), UOp.const((0.0, 0.0), dtypes.half)) for i in range(2)]
  vsf = [_load2(Vs_raw, (ku0 + m_lane) * K_STR + n_off_u + j * 8 + n_base) for j in range(4)]
  for i in range(2):
    for j in range(4): cu[i][j] = UOp.wmma(af_u[i], vsf[j], cu[i][j], (8, 8, 8), 32)
  uf = u_acc.after(UOp.group(*[u_acc[i, j, e].store(cu[i][j][e]) for i in range(2) for j in range(4) for e in range(2)]).end(kk_u))
  st_u = [U_out.after(b_vs)[h, c, wm * 16 + i * 8 + m_lane, n_off_u + j * 8 + n_base + e].store(uf[i, j, e].load().cast(dtypes.half))
          for i in range(2) for j in range(4) for e in range(2)]
  return UOp.group(*st_u).end(nb_u).end(lane, wid, hc).sink(arg=KernelInfo(name="fused_conv_wy_c32", opts_to_apply=()))

def _build_chunk_scan(H:int, N:int, NC:int, K:int=128, V:int=128, BV:int=16,
                      has_init:bool=False, save_out:bool=False) -> UOp:
  C = CHUNK_SIZE
  VT = V // BV
  assert K == 128 and BV == 16 and V % BV == 0 and C == 32
  out = UOp.param(0, dtypes.float32, (H, N, V))
  W_in = UOp.param(1, dtypes.half, (H, NC, C, K))
  U_in = UOp.param(2, dtypes.half, (H, NC, C, V))
  Qg_in = UOp.param(3, dtypes.half, (H, NC, C, K))
  Kg_in = UOp.param(4, dtypes.half, (H, NC, C, K))
  L_in = UOp.param(5, dtypes.half, (H, NC, C, C))
  gl_in = UOp.param(6, dtypes.float32, (H, NC))
  ctok = UOp.param(7, dtypes.int32, (NC, C))
  cflags = UOp.param(8, dtypes.int32, (NC,))
  p_idx = 9
  init_S = UOp.param(p_idx, dtypes.half, (H, V, K)) if has_init else None
  if has_init: p_idx += 1
  state_out = UOp.param(p_idx, dtypes.half, (H, V, K)) if save_out else None

  h_vt = UOp.range(H * VT, 0, AxisType.GLOBAL)
  h, vt = h_vt // VT, h_vt % VT
  v_base = vt * BV
  wk = UOp.range(4, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  tid = wk * 32 + lane

  m_lane = ((lane >> 1) & 3) + ((lane >> 4) * 4)
  n_base = ((lane & 1) * 2) + (((lane >> 3) & 1) * 4)

  S_reg = UOp.alloc((2, 4, 2), dtypes.float32, addrspace=AddrSpace.REG)
  S_reg = S_reg.after(*[S_reg[i, j, e].store(0.0) for i in range(2) for j in range(4) for e in range(2)])

  # 13.5 KB LDS -> 2 TGs/core
  U_STR = BV + 4
  saved_S = UOp.alloc((BV, K), dtypes.half, addrspace=AddrSpace.LOCAL)
  S_lds = UOp.alloc((BV, K + 4), dtypes.half, addrspace=AddrSpace.LOCAL)
  part_lds = UOp.alloc((4, C, BV + 2), dtypes.half, addrspace=AddrSpace.LOCAL)
  Unew_raw = UOp.alloc((C * U_STR,), dtypes.half, addrspace=AddrSpace.LOCAL)
  Unew_lds = Unew_raw.reshape(C, U_STR)

  zi = UOp.range(16, 5)
  init_v = init_S[h, v_base + zi, tid].load() if has_init else UOp.const(0.0, dtypes.half)
  saved_S = saved_S.after(saved_S[zi, tid].store(init_v).end(zi).barrier())

  c = UOp.range(NC, 100)
  flag = cflags[c].load()
  is_restore, is_save = flag.eq(2), flag.eq(1)
  g_last = gl_in[h, c].load()
  S_prev, sav_prev = S_reg.after(c), saved_S.after(c)

  s_cur = [[[None]*2 for _ in range(4)] for _ in range(2)]
  st_slds = []
  for i in range(2):
    rv = i * 8 + m_lane
    for j in range(4):
      for e in range(2):
        ck = wk * 32 + j * 8 + n_base + e
        val = is_restore.where(sav_prev[rv, ck].load().float(), S_prev[i, j, e].load())
        s_cur[i][j][e] = val
        st_slds.append(S_lds.after(c)[rv, ck].store(val.cast(dtypes.half)))
  S_h = S_lds.after(UOp.group(*st_slds).barrier())

  W_flat, Qg_flat, Kg_flat, L_flat = W_in.flatten(), Qg_in.flatten(), Kg_in.flatten(), L_in.flatten()
  hc_off_k = (h * NC + c) * C * K
  hc_off_c = (h * NC + c) * C * C
  # Split ws and qs into two phases to cut peak live accumulator registers from 48 -> 32 floats!
  ws = [[UOp.const((0.0, 0.0), dtypes.float32) for _ in range(2)] for _ in range(4)]
  for kk in range(4):
    k0 = wk * 32 + kk * 8
    sf_b = [UOp.stack(S_h[j * 8 + n_base, k0 + m_lane].load(), S_h[j * 8 + n_base + 1, k0 + m_lane].load()) for j in range(2)]
    for i in range(4):
      wf_a = _load2(W_flat, hc_off_k + (i * 8 + m_lane) * K + k0 + n_base)
      for j in range(2): ws[i][j] = UOp.wmma(wf_a, sf_b[j], ws[i][j], (8, 8, 8), 32)

  st_ws = [part_lds.after(c)[wk, i * 8 + m_lane, j * 8 + n_base + e].store(ws[i][j][e].cast(dtypes.half))
           for i in range(4) for j in range(2) for e in range(2)]
  p_ws = part_lds.after(UOp.group(*st_ws).barrier())

  ri_u = UOp.range(4, 110)
  idx_u = ri_u * 128 + tid
  tu, vu = idx_u // 16, idx_u % 16
  ws_sum = p_ws[0, tu, vu].load().float() + p_ws[1, tu, vu].load().float() + p_ws[2, tu, vu].load().float() + p_ws[3, tu, vu].load().float()
  st_unew = Unew_lds.after(c)[tu, vu].store((U_in[h, c, tu, v_base + vu].load().float() - ws_sum).cast(dtypes.half)).end(ri_u).barrier()

  qs = [[UOp.const((0.0, 0.0), dtypes.float32) for _ in range(2)] for _ in range(4)]
  for kk in range(4):
    k0 = wk * 32 + kk * 8
    sf_b = [UOp.stack(S_h[j * 8 + n_base, k0 + m_lane].load(), S_h[j * 8 + n_base + 1, k0 + m_lane].load()) for j in range(2)]
    for i in range(4):
      qf_a = _load2(Qg_flat, hc_off_k + (i * 8 + m_lane) * K + k0 + n_base)
      for j in range(2): qs[i][j] = UOp.wmma(qf_a, sf_b[j], qs[i][j], (8, 8, 8), 32)
  st_qs = [part_lds.after(st_unew)[wk, i * 8 + m_lane, j * 8 + n_base + e].store(qs[i][j][e].cast(dtypes.half))
           for i in range(4) for j in range(2) for e in range(2)]
  p_qs = part_lds.after(UOp.group(*st_qs).barrier())
  Un_r, Un_raw_r = Unew_lds.after(st_unew), Unew_raw.after(st_unew)

  oi_acc = [UOp.const((0.0, 0.0), dtypes.float32) for _ in range(2)]
  rm_o = wk * 8 + m_lane
  for kk in range(4):
    k0 = kk * 8
    lf_a = _load2(L_flat, hc_off_c + rm_o * C + k0 + n_base)
    for j in range(2):
      uf_b = _load2(Un_raw_r, (k0 + m_lane) * U_STR + j * 8 + n_base)
      oi_acc[j] = UOp.wmma(lf_a, uf_b, oi_acc[j], (8, 8, 8), 32)

  tok_o = ctok[c, rm_o].load()
  vo, tok_os = (tok_o >= 0) & (tok_o < N), tok_o.maximum(0).minimum(N - 1)
  st_o = []
  for j in range(2):
    for e in range(2):
      cv = j * 8 + n_base + e
      o_inter = p_qs[0, rm_o, cv].load().float() + p_qs[1, rm_o, cv].load().float() + p_qs[2, rm_o, cv].load().float() + p_qs[3, rm_o, cv].load().float()
      st_o.append(out[h, tok_os.valid(vo), v_base + cv].store(o_inter + oi_acc[j][e]))

  sn = [[UOp.stack(s_cur[i][j][0] * g_last, s_cur[i][j][1] * g_last) for j in range(4)] for i in range(2)]
  for kk in range(4):
    k0 = kk * 8
    ut_a = [UOp.stack(Un_r[k0 + n_base, i * 8 + m_lane].load(), Un_r[k0 + n_base + 1, i * 8 + m_lane].load()) for i in range(2)]
    kg_b = [_load2(Kg_flat, hc_off_k + (k0 + m_lane) * K + wk * 32 + j * 8 + n_base) for j in range(4)]
    for i in range(2):
      for j in range(4): sn[i][j] = UOp.wmma(ut_a[i], kg_b[j], sn[i][j], (8, 8, 8), 32)

  st_s = []
  for i in range(2):
    rv = i * 8 + m_lane
    for j in range(4):
      for e in range(2):
        ck = wk * 32 + j * 8 + n_base + e
        st_s += [S_reg[i, j, e].store(sn[i][j][e]),
                 saved_S[rv.valid(is_save), ck].store(sn[i][j][e].cast(dtypes.half))]
  loop_done = UOp.group(*st_o, *st_s).barrier().end(c)
  if save_out:
    zo = UOp.range(16, 6)
    sav_f = saved_S.after(loop_done)
    st_out_s = state_out[h, v_base + zo, tid].store(sav_f[zo, tid].load()).end(zo)
    body = UOp.group(loop_done, st_out_s)
  else:
    body = loop_done
  sfx = ("_i" if has_init else "") + ("_s" if save_out else "")
  return body.end(lane, wk, h_vt).sink(arg=KernelInfo(name=f"chunk_scan_c32{sfx}", opts_to_apply=()))

@functools.cache
def get_fused_conv_wy(H:int, N:int, NC:int, P:int, K:int, V:int,
                      q_off:int, k_off:int, v_off:int, beta_off:int, alpha_off:int, has_prefix:bool=False) -> UOp:
  sfx = "_p" if has_prefix else ""
  return _emit(f"fused_conv_wy_{H}_{N}_{NC}_{P}{sfx}",
               _build_fused_conv_wy(H, N, NC, P, K, V, q_off, k_off, v_off, beta_off, alpha_off, has_prefix))

@functools.cache
def get_chunk_scan(H:int, N:int, NC:int, K:int, V:int, has_init:bool=False, save_out:bool=False) -> UOp:
  sfx = ("_i" if has_init else "") + ("_s" if save_out else "")
  return _emit(f"chunk_scan_{H}_{N}_{NC}_{K}_{V}{sfx}", _build_chunk_scan(H, N, NC, K, V, 16, has_init, save_out))

def fused_delta_core(blk, p:Tensor, ctok:Tensor, cflags:Tensor, csrc:Tensor,
                     conv_prefix:Tensor|None=None, init_state:Tensor|None=None,
                     return_state:bool=False) -> Tensor|tuple[Tensor, Tensor]|None:
  """Run fused conv+norm+WY+WMMA chunk scan on fused projection `p (N, P)` -> `core (H, N, V)` f32, or None.

  When `return_state=True`, returns `(core, saved_state (H, V, K) f16)` for prefix caching."""
  if not enabled() or not _is_metal(p.device) or p.dtype is not dtypes.float16: return None
  cfg = blk.config.ssm
  H, K, V = blk.num_v_heads, blk.head_k_dim, blk.head_v_dim
  if cfg.kda or cfg.gate_lower_bound is not None or blk.num_k_heads != H or K != 128 or V != 128: return None
  w_conv = blk.ssm_conv1d["weight"].flatten(1)
  if w_conv.shape[1] != 4 or w_conv.dtype is not dtypes.float16: return None
  N, P = p.shape[-2], p.shape[-1]
  NC = ctok.shape[0]
  qa, qb, qc = blk._proj_ends
  q_off, k_off, v_off = 0, blk.q_dim, 2 * blk.q_dim
  if v_off + H * V != qa: return None
  has_pref, has_init = conv_prefix is not None, init_state is not None
  k_prep = get_fused_conv_wy(H, N, NC, P, K, V, q_off, k_off, v_off, qb, qc, has_pref)
  k_scan = get_chunk_scan(H, N, NC, K, V, has_init, return_state)
  dev = p.device
  C = CHUNK_SIZE
  W, U = UOp.alloc((H, NC, C, K), dtypes.half, device=dev), UOp.alloc((H, NC, C, V), dtypes.half, device=dev)
  Qg, Kg = UOp.alloc((H, NC, C, K), dtypes.half, device=dev), UOp.alloc((H, NC, C, K), dtypes.half, device=dev)
  L, gl = UOp.alloc((H, NC, C, C), dtypes.half, device=dev), UOp.alloc((H, NC), dtypes.float32, device=dev)
  prep_args = [W, U, Qg, Kg, L, gl, p.reshape(N, P).contiguous().uop, w_conv.contiguous().uop,
               blk.ssm_dt["bias"].reshape(H).float().contiguous().uop, blk.ssm_a.reshape(H).float().contiguous().uop,
               ctok.uop, csrc.uop]
  if has_pref: prep_args.append(conv_prefix.contiguous().uop)
  c1 = k_prep.call(*prep_args)
  out = UOp.alloc((H, N, V), dtypes.float32, device=dev)
  scan_args = [out, W.after(c1), U.after(c1), Qg.after(c1), Kg.after(c1), L.after(c1), gl.after(c1), ctok.uop, cflags.uop]
  if has_init: scan_args.append(init_state.contiguous().uop)
  if return_state:
    s_out = UOp.alloc((H, V, K), dtypes.half, device=dev)
    scan_args.append(s_out)
    c2 = k_scan.call(*scan_args)
    return Tensor(out.after(c2)), Tensor(s_out.after(c2))
  c2 = k_scan.call(*scan_args)
  return Tensor(out.after(c2))
