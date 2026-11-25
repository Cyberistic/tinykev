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

def _build_fused_conv_wy(H:int, N:int, NC:int, P:int, K:int, V:int,
                         q_off:int, k_off:int, v_off:int, beta_off:int, alpha_off:int) -> UOp:
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

  hc = UOp.range(H * NC, 0, AxisType.GLOBAL)
  h, c = hc // NC, hc % NC
  wid = UOp.range(4, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  wm, wn = wid // 2, wid % 2
  tid = wid * 32 + lane

  m_lane = ((lane >> 1) & 3) + ((lane >> 4) * 4)
  n_base = ((lane & 1) * 2) + (((lane >> 3) & 1) * 4)

  # < 14 KB LDS total -> 2 threadgroups/core on Apple Silicon
  K_STR, C_STR = K + 4, C + 4
  K_raw = UOp.alloc((C * K_STR,), dtypes.half, addrspace=AddrSpace.LOCAL)
  M_raw = UOp.alloc((C * C_STR,), dtypes.half, addrspace=AddrSpace.LOCAL)
  A_raw = UOp.alloc((C * C_STR,), dtypes.half, addrspace=AddrSpace.LOCAL)
  K_lds, M_lds, A_lds = K_raw.reshape(C, K_STR), M_raw.reshape(C, C_STR), A_raw.reshape(C, C_STR)
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

  # Preload w_conv for this lane's 4 channels (Q, K, V) x 4 taps before the 8-step loop
  wq = [[w_conv[q_off + h * K + lane * 4 + e, j].load().float() for j in range(4)] for e in range(4)]
  wk = [[w_conv[k_off + h * K + lane * 4 + e, j].load().float() for j in range(4)] for e in range(4)]
  wv = [[w_conv[v_off + h * V + lane * 4 + e, j].load().float() for j in range(4)] for e in range(4)]

  # 2. 8-step zero-barrier loop: each warp handles 8 tokens, each lane owns 4 contiguous channels -> warp simd_sum L2 norm!
  step = UOp.range(8, 10)
  t_cur = step * 4 + wid
  tok_r = ctok[c, t_cur].load()
  vr = (tok_r >= 0) & (tok_r < N)
  tok_rs = tok_r.maximum(0).minimum(N - 1)
  s_idx = [csrc[tok_rs, j].load() for j in range(3)]
  s_val = [(idx >= 0) & (idx < N) for idx in s_idx]
  s_safe = [idx.maximum(0).minimum(N - 1) for idx in s_idx]

  q_vals, k_vals, v_vals = [], [], []
  q_sq_l, k_sq_l = UOp.const(0.0, dtypes.float32), UOp.const(0.0, dtypes.float32)
  for e in range(4):
    ch = lane * 4 + e
    qc, kc_ch, vc = q_off + h * K + ch, k_off + h * K + ch, v_off + h * V + ch
    qa = p_in[tok_rs, qc].load().float() * wq[e][3]
    ka = p_in[tok_rs, kc_ch].load().float() * wk[e][3]
    va = p_in[tok_rs, vc].load().float() * wv[e][3]
    for j in range(3):
      qa = qa + s_val[j].where(p_in[s_safe[j], qc].load().float(), 0.0) * wq[e][j]
      ka = ka + s_val[j].where(p_in[s_safe[j], kc_ch].load().float(), 0.0) * wk[e][j]
      va = va + s_val[j].where(p_in[s_safe[j], vc].load().float(), 0.0) * wv[e][j]
    qv0, kv0, vv0 = vr.where(_silu(qa), 0.0), vr.where(_silu(ka), 0.0), vr.where(_silu(va), 0.0)
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
        st_ml += [M_lds[ri, cj].store(mv.cast(dtypes.half)), L_out[h, c, ri, cj].store(lv.cast(dtypes.half)),
                  A_lds[ri, cj].store(UOp.const(0.0, dtypes.half))]
  b_ml = UOp.group(*st_ml).barrier()
  M_r, M_raw_r = M_lds.after(b_ml), M_raw.after(b_ml)

  # 4a. Parallel 2x16 block-diagonal inversion in warp 0 (lanes 0..15 -> A00, lanes 16..31 -> A11; zero barriers!)
  blk, j16 = lane // 16, lane % 16
  base16 = blk * 16
  a_col = [UOp.const(0.0, dtypes.float32) for _ in range(16)]
  st_diag = []
  for i in range(16):
    if i == 0:
      val = j16.eq(0).where(UOp.const(1.0, dtypes.float32), UOp.const(0.0, dtypes.float32))
    else:
      acc_s = UOp.const(0.0, dtypes.float32)
      for m in range(i): acc_s = acc_s + M_r[base16 + i, base16 + m].load().float() * a_col[m]
      val = j16.eq(i).where(UOp.const(1.0, dtypes.float32), (j16 < i).where(-acc_s, UOp.const(0.0, dtypes.float32)))
    a_col[i] = val
    st_diag.append(A_lds.after(b_ml)[(base16 + i).valid(wid.eq(0)), base16 + j16].store(val.cast(dtypes.half)))

  si = UOp.range(C, 40)
  scale_k = b_r[si].load() * (g_r[si].load() * LOG2E).exp2()
  st_ks = K_lds.after(b_ml)[si, tid].store((K_r[si, tid].load().float() * scale_k).cast(dtypes.half)).end(si)
  b_diag = UOp.group(*st_diag, st_ks).barrier()
  A_d_raw, Ks_raw = A_raw.after(b_diag), K_raw.after(b_diag)

  # 4b. Off-diagonal block A10 = -A11 @ M10 @ A00 via UOp.wmma
  rm16, cn16 = wm * 8 + m_lane, wn * 8 + n_base
  t_acc = UOp.const((0.0, 0.0), dtypes.float32)
  for kk in range(2):
    k0 = kk * 8
    mf_a = _load2(M_raw_r, (16 + rm16) * C_STR + k0 + n_base)
    af_b = _load2(A_d_raw, (k0 + m_lane) * C_STR + cn16)
    t_acc = UOp.wmma(mf_a, af_b, t_acc, (8, 8, 8), 32)
  b_t = UOp.group(M_lds.after(b_diag)[rm16, 16 + cn16].store(t_acc[0].cast(dtypes.half)),
                  M_lds.after(b_diag)[rm16, 16 + cn16 + 1].store(t_acc[1].cast(dtypes.half))).barrier()
  T_raw = M_raw.after(b_t)

  a10 = UOp.const((0.0, 0.0), dtypes.float32)
  for kk in range(2):
    k0 = kk * 8
    a11_a = _load2(A_d_raw, (16 + rm16) * C_STR + 16 + k0 + n_base)
    t_b = _load2(T_raw, (k0 + m_lane) * C_STR + 16 + cn16)
    a10 = UOp.wmma(a11_a, t_b, a10, (8, 8, 8), 32)
  b_a = UOp.group(A_lds.after(b_t)[16 + rm16, cn16].store((-a10[0]).cast(dtypes.half)),
                  A_lds.after(b_t)[16 + rm16, cn16 + 1].store((-a10[1]).cast(dtypes.half))).barrier()
  A_raw_r = A_raw.after(b_a)

  # 5. Looped W = A @ Ks -> W_out, reload Vs -> K_lds, looped U = A @ Vs -> U_out
  nb_w = UOp.range(2, 50)
  n_off_w = (wn * 2 + nb_w) * 32
  w_acc = UOp.alloc((2, 4, 2), dtypes.float32, addrspace=AddrSpace.REG)
  w_acc = w_acc.after(*[w_acc.after(nb_w)[i, j, e].store(0.0) for i in range(2) for j in range(4) for e in range(2)])
  kk_w = UOp.range(4, 51)
  kw0 = kk_w * 8
  cw = [[UOp.stack(w_acc.after(kk_w)[i, j, 0].load(), w_acc.after(kk_w)[i, j, 1].load()) for j in range(4)] for i in range(2)]
  af_w = [_load2(A_raw_r, (wm * 16 + i * 8 + m_lane) * C_STR + kw0 + n_base) for i in range(2)]
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
  cu = [[UOp.stack(u_acc.after(kk_u)[i, j, 0].load(), u_acc.after(kk_u)[i, j, 1].load()) for j in range(4)] for i in range(2)]
  af_u = [_load2(A_raw_r, (wm * 16 + i * 8 + m_lane) * C_STR + ku0 + n_base) for i in range(2)]
  vsf = [_load2(Vs_raw, (ku0 + m_lane) * K_STR + n_off_u + j * 8 + n_base) for j in range(4)]
  for i in range(2):
    for j in range(4): cu[i][j] = UOp.wmma(af_u[i], vsf[j], cu[i][j], (8, 8, 8), 32)
  uf = u_acc.after(UOp.group(*[u_acc[i, j, e].store(cu[i][j][e]) for i in range(2) for j in range(4) for e in range(2)]).end(kk_u))
  st_u = [U_out.after(b_vs)[h, c, wm * 16 + i * 8 + m_lane, n_off_u + j * 8 + n_base + e].store(uf[i, j, e].load().cast(dtypes.half))
          for i in range(2) for j in range(4) for e in range(2)]
  return UOp.group(*st_u).end(nb_u).end(lane, wid, hc).sink(arg=KernelInfo(name="fused_conv_wy_c32", opts_to_apply=()))

def _build_chunk_scan(H:int, N:int, NC:int, K:int=128, V:int=128, BV:int=16) -> UOp:
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
  saved_S = saved_S.after(saved_S[zi, tid].store(UOp.const(0.0, dtypes.half)).end(zi).barrier())

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
  ws = [[UOp.const((0.0, 0.0), dtypes.float32) for _ in range(2)] for _ in range(4)]
  qs = [[UOp.const((0.0, 0.0), dtypes.float32) for _ in range(2)] for _ in range(4)]
  for kk in range(4):
    k0 = wk * 32 + kk * 8
    sf_b = [UOp.stack(S_h[j * 8 + n_base, k0 + m_lane].load(), S_h[j * 8 + n_base + 1, k0 + m_lane].load()) for j in range(2)]
    for i in range(4):
      rm = i * 8 + m_lane
      wf_a = _load2(W_flat, hc_off_k + rm * K + k0 + n_base)
      qf_a = _load2(Qg_flat, hc_off_k + rm * K + k0 + n_base)
      for j in range(2):
        ws[i][j] = UOp.wmma(wf_a, sf_b[j], ws[i][j], (8, 8, 8), 32)
        qs[i][j] = UOp.wmma(qf_a, sf_b[j], qs[i][j], (8, 8, 8), 32)

  st_ws = [part_lds.after(c)[wk, i * 8 + m_lane, j * 8 + n_base + e].store(ws[i][j][e].cast(dtypes.half))
           for i in range(4) for j in range(2) for e in range(2)]
  p_ws = part_lds.after(UOp.group(*st_ws).barrier())

  ri_u = UOp.range(4, 110)
  idx_u = ri_u * 128 + tid
  tu, vu = idx_u // 16, idx_u % 16
  ws_sum = p_ws[0, tu, vu].load().float() + p_ws[1, tu, vu].load().float() + p_ws[2, tu, vu].load().float() + p_ws[3, tu, vu].load().float()
  st_unew = Unew_lds.after(c)[tu, vu].store((U_in[h, c, tu, v_base + vu].load().float() - ws_sum).cast(dtypes.half)).end(ri_u).barrier()

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
  return UOp.group(*st_o, *st_s).barrier().end(c).end(lane, wk, h_vt).sink(arg=KernelInfo(name="chunk_scan_c32", opts_to_apply=()))

@functools.cache
def get_fused_conv_wy(H:int, N:int, NC:int, P:int, K:int, V:int,
                      q_off:int, k_off:int, v_off:int, beta_off:int, alpha_off:int) -> UOp:
  return _emit(f"fused_conv_wy_{H}_{N}_{NC}_{P}",
               _build_fused_conv_wy(H, N, NC, P, K, V, q_off, k_off, v_off, beta_off, alpha_off))

@functools.cache
def get_chunk_scan(H:int, N:int, NC:int, K:int, V:int) -> UOp:
  return _emit(f"chunk_scan_{H}_{N}_{NC}_{K}_{V}", _build_chunk_scan(H, N, NC, K, V))

def fused_delta_core(blk, p:Tensor, ctok:Tensor, cflags:Tensor, csrc:Tensor) -> Tensor|None:
  """Run fused conv+norm+WY+WMMA chunk scan on fused projection `p (N, P)` -> `core (H, N, V)` f32, or None."""
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
  k_prep = get_fused_conv_wy(H, N, NC, P, K, V, q_off, k_off, v_off, qb, qc)
  k_scan = get_chunk_scan(H, N, NC, K, V)
  dev = p.device
  C = CHUNK_SIZE
  W, U = UOp.alloc((H, NC, C, K), dtypes.half, device=dev), UOp.alloc((H, NC, C, V), dtypes.half, device=dev)
  Qg, Kg = UOp.alloc((H, NC, C, K), dtypes.half, device=dev), UOp.alloc((H, NC, C, K), dtypes.half, device=dev)
  L, gl = UOp.alloc((H, NC, C, C), dtypes.half, device=dev), UOp.alloc((H, NC), dtypes.float32, device=dev)
  c1 = k_prep.call(W, U, Qg, Kg, L, gl, p.reshape(N, P).contiguous().uop, w_conv.contiguous().uop,
                   blk.ssm_dt["bias"].reshape(H).float().contiguous().uop, blk.ssm_a.reshape(H).float().contiguous().uop,
                   ctok.uop, csrc.uop)
  out = UOp.alloc((H, N, V), dtypes.float32, device=dev)
  c2 = k_scan.call(out, W.after(c1), U.after(c1), Qg.after(c1), Kg.after(c1), L.after(c1), gl.after(c1), ctok.uop, cflags.uop)
  return Tensor(out.after(c2))
