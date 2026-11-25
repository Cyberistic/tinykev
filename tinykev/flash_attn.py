"""Packed Metal UIR Flash Attention: fused QK-RMSNorm+RoPE and block-causal online-softmax attention.

Replaces the 14 per-layer attention kernels (4 row_gathers, q/k RMSNorm reduce+scale, RoPE, QK^T, 2x softmax,
P@V, and sigmoid output gate) with two zero-LDS, zero-barrier warp kernels operating directly on the packed
`q|k|v` projection `p [N, P]` using `(rows_idx, tok_row, tok_pos)`.
"""
from __future__ import annotations

import functools, math, os, pathlib, threading

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
LOG2E = math.log2(math.e)

def enabled() -> bool: return os.environ.get("TINYKEV_FLASH_ATTN", "1") != "0"

def _is_metal(device) -> bool:
  d = device if isinstance(device, str) else (device[0] if isinstance(device, tuple) and device else None)
  return isinstance(d, str) and d.split(":")[0] == "METAL"

def _simd_sum(v:UOp) -> UOp:
  if v.op is Ops.REDUCE or any(u.op is Ops.REDUCE for u in v.src):
    tmp = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
    v = tmp.after(tmp[0].store(v))[0].load()
  return UOp.native("simd_sum({0})", shape=v.shape, dtype=dtypes.float32, fmt=True).call(v).reshape(())

def _emit(key:str, u:UOp) -> UOp:
  with Context(SPEC=0):
    uir = render_uir(graph_rewrite(u, symbolic_simple, enter_calls=True), color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"{key}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def _build_qk_norm_rope(N:int, nh:int, nkv:int, hd:int, rope_dim:int, P:int, k_off:int,
                        gated:bool, has_norm:bool, eps:float, max_pos:int) -> UOp:
  """Fused Q/K RMSNorm + partial NeoX RoPE in packed order, reading fused `p [N, P]` directly.

  Lane `l in 0..31` owns `d = l + c*32` for `c = 0..hd//32 - 1`.  When `rope_dim == 64`, `c=0` (`0..31`)
  and `c=1` (`32..63`) live in the exact same lane, so RoPE needs zero cross-lane traffic and zero LDS."""
  assert hd % 32 == 0 and rope_dim == 64 and hd >= 64
  E = hd // 32
  q_out = UOp.param(0, dtypes.half, (N, nh, hd))
  k_out = UOp.param(1, dtypes.half, (N, nkv, hd))
  p = UOp.param(2, dtypes.half, (N, P))
  qw = UOp.param(3, dtypes.half, (hd,))
  kw = UOp.param(4, dtypes.half, (hd,))
  freqs = UOp.param(5, dtypes.float32, (max_pos, rope_dim))
  tok_pos = UOp.param(6, dtypes.int32, (N,))

  H_tot = nh + nkv
  th = UOp.range(N * H_tot, 0, AxisType.GLOBAL)
  t, h_all = th // H_tot, th % H_tot
  is_q = h_all < nh
  h_q = h_all.minimum(nh - 1)
  h_k = (h_all - nh).maximum(0)
  lane = UOp.range(32, -1, AxisType.WARP)

  q_stride = 2 * hd if gated else hd
  base_off = is_q.where(h_q * q_stride, k_off + h_k * hd)

  # Load E elements per lane (`d = lane + c * 32`)
  reg = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  c_r = UOp.range(E, 2, AxisType.UPCAST)
  d = lane + c_r * 32
  reg = reg.after(reg[c_r].store(p[t, base_off + d].load().float()).end(c_r))

  if has_norm:
    sq = _simd_sum((reg[c_r].load() * reg[c_r].load()).reduce(c_r, arg=(Ops.ADD, 0)))
    inv = (sq * (1.0 / hd) + eps).rsqrt()
    vals = []
    for c in range(E):
      dc = lane + c * 32
      w = is_q.where(qw[dc].load(), kw[dc].load())
      vals.append(((reg[c].load() * inv).cast(dtypes.half) * w).float())
  else:
    vals = [reg[c].load() for c in range(E)]

  pos = tok_pos[t].load().maximum(0).minimum(max_pos - 1)
  cos = freqs[pos, lane].load()
  sin = freqs[pos, lane + 32].load()
  y0 = vals[0] * cos - vals[1] * sin
  y1 = vals[1] * cos + vals[0] * sin
  out_vals = [y0, y1] + vals[2:]

  sts = []
  for c in range(E):
    dc = lane + c * 32
    vh = out_vals[c].cast(dtypes.half)
    sts.append(q_out[t, h_q.valid(is_q), dc].store(vh))
    sts.append(k_out[t, h_k.valid(is_q.eq(False)), dc].store(vh))
  return UOp.group(*sts).end(lane, th).sink(arg=KernelInfo(name=f"packed_qk_norm_rope_{N}_{nh}_{nkv}", opts_to_apply=()))

def _build_packed_flash_attn(N:int, R:int, L:int, nh:int, nkv:int, hd:int, P:int, v_off:int, gated:bool) -> UOp:
  """Zero-LDS, zero-barrier packed block-causal Flash Attention + fused sigmoid output gate.

  1 warp (32 lanes) per `(t, h)`; each lane holds `hd//32` contiguous elements (`lane*E + e`) of `q` and `acc`
  in registers, walks only `j = 0 .. tok_pos[t]` via `rows_idx[tok_row[t]*L + j]`, and fuses `* sigmoid(gate)`."""
  assert hd % 32 == 0 and nh % nkv == 0
  E = hd // 32
  gqa = nh // nkv
  scale = hd ** -0.5

  out = UOp.param(0, dtypes.half, (N, nh * hd))
  q_in = UOp.param(1, dtypes.half, (N, nh, hd))
  k_in = UOp.param(2, dtypes.half, (N, nkv, hd))
  p_in = UOp.param(3, dtypes.half, (N, P))
  rows_idx = UOp.param(4, dtypes.int32, (R * L,))
  tok_row = UOp.param(5, dtypes.int32, (N,))
  tok_pos = UOp.param(6, dtypes.int32, (N,))

  th = UOp.range(N * nh, 0, AxisType.GLOBAL)
  t, h = th // nh, th % nh
  kh = h // gqa
  lane = UOp.range(32, -1, AxisType.WARP)

  # Preload q[t, h, lane*E .. lane*E+E-1] * scale into registers
  q_reg = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  e_u = UOp.range(E, 2, AxisType.UPCAST)
  d = lane * E + e_u
  q_reg = q_reg.after(q_reg[e_u].store(q_in[t, h, d].load().float() * scale).end(e_u))

  acc = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  m_reg = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  l_reg = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[e_u].store(0.0).end(e_u))
  m_reg = m_reg.after(m_reg[0].store(UOp.const(-1e30, dtypes.float32)))
  l_reg = l_reg.after(l_reg[0].store(0.0))

  row_base = tok_row[t].load().maximum(0).minimum(R - 1) * L
  pos_t = tok_pos[t].load().maximum(0).minimum(L - 1)

  j = UOp.range(L, 10)
  valid_j = j <= pos_t
  kj = rows_idx[(row_base + j).valid(valid_j)].load().maximum(0).minimum(N - 1)

  dot_l = (q_reg[e_u].load() * k_in[kj.valid(valid_j), kh, d].load().float()).reduce(e_u, arg=(Ops.ADD, 0))
  score = valid_j.where(_simd_sum(dot_l), UOp.const(-1e30, dtypes.float32))

  m_prev = m_reg.after(j)[0].load()
  l_prev = l_reg.after(j)[0].load()
  m_new = m_prev.maximum(score)
  alpha = ((m_prev - m_new) * LOG2E).exp2()
  w = valid_j.where(((score - m_new) * LOG2E).exp2(), UOp.const(0.0, dtypes.float32))

  v_off_kh = v_off + kh * hd
  acc_prev = acc.after(j)
  upd = UOp.group(
    m_reg[0].store(m_new),
    l_reg[0].store(l_prev * alpha + w),
    acc[e_u].store(acc_prev[e_u].load() * alpha + w * p_in[kj.valid(valid_j), v_off_kh + d].load().float()).end(e_u),
  ).end(j)

  inv_l = l_reg.after(upd)[0].load().reciprocal()
  acc_f = acc.after(upd)
  e_o = UOp.range(E, 3, AxisType.UPCAST)
  d_o = lane * E + e_o
  val = acc_f[e_o].load() * inv_l
  if gated:
    gv = p_in[t, h * 2 * hd + hd + d_o].load().float()
    val = val * ((-gv * LOG2E).exp2() + 1.0).reciprocal()
  return out[t, h * hd + d_o].store(val.cast(dtypes.half)).end(e_o).end(lane, th).sink(
    arg=KernelInfo(name=f"packed_flash_attn_{N}_{R}_{L}_{nh}_{nkv}", opts_to_apply=()))

def make_q_tiles(S:int, branches:list[list[int]], L:int, BM:int=16, prefix_cached:bool=False) -> list[list[int]]:
  """List of `[row_r, q_start, q_len]` tiles covering every active packed token once."""
  tiles: list[list[int]] = []
  for r, b in enumerate(branches):
    row_len = S + len(b)
    cur = S if (prefix_cached or r > 0) else 0
    while cur < row_len:
      ln = min(BM, row_len - cur)
      tiles.append([r, cur, ln])
      cur += ln
  return tiles or [[0, 0, 1]]

def _load2(buf_flat:UOp, off:UOp) -> UOp:
  return UOp(Ops.SHRINK, src=(buf_flat, off, UOp.const(2))).load()

def _build_packed_flash_attn_wmma(N:int, R:int, L:int, NT:int, nh:int, nkv:int, hd:int, P:int, v_off:int,
                                  gated:bool, S_p:int=0) -> UOp:
  """WMMA-accelerated packed block-causal Flash Attention (< 14 KB LDS -> 2 TGs/core on Apple Silicon).

  Grid: `NT * nh` threadgroups; Local: 4 warps (128 threads), each warp owns 64 dims of `Q` and `O`.
  Both `S = Q @ K^T` and `O += P @ V` run on Metal 8x8x8 tensor cores (`UOp.wmma`), with 16-lane
  `simd_shuffle_xor` online softmax and fused `* sigmoid(gate)` on output store.
  When `S_p > 0` (cached-prefix mode), reads `k_pref`/`v_pref` directly for `rows_idx < S_p` with zero `cat`."""
  BM, BN = 16, 16
  assert hd == 256 and nh % nkv == 0
  gqa = nh // nkv
  scale = hd ** -0.5

  out = UOp.param(0, dtypes.half, (N, nh * hd))
  q_in = UOp.param(1, dtypes.half, (N, nh, hd))
  k_in = UOp.param(2, dtypes.half, (N, nkv, hd))
  p_in = UOp.param(3, dtypes.half, (N, P))
  rows_idx = UOp.param(4, dtypes.int32, (R * L,))
  q_tiles = UOp.param(5, dtypes.int32, (NT, 3))
  k_pref = UOp.param(6, dtypes.half, (S_p, nkv, hd)) if S_p > 0 else None
  v_pref = UOp.param(7, dtypes.half, (S_p, nkv * hd)) if S_p > 0 else None

  th = UOp.range(NT * nh, 0, AxisType.GLOBAL)
  ti, h = th // nh, th % nh
  kh = h // gqa
  wid = UOp.range(4, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  tid = wid * 32 + lane

  m_lane = ((lane >> 1) & 3) + ((lane >> 4) * 4)
  n_base = ((lane & 1) * 2) + (((lane >> 3) & 1) * 4)

  row_r = q_tiles[ti, 0].load().maximum(0).minimum(R - 1)
  q_start = q_tiles[ti, 1].load().maximum(0).minimum(L - 1)
  q_len = q_tiles[ti, 2].load().maximum(1).minimum(BM)
  row_base = row_r * L

  # LDS: KV(8.32 KB) + S_part(4.61 KB) + P(0.64 KB) + m/l/a(0.19 KB) = 13.76 KB (< 16 KB, 4 barriers/tile)
  D_STR, P_STR = hd + 4, BN + 4
  KV_raw = UOp.alloc((BN * D_STR,), dtypes.half, addrspace=AddrSpace.LOCAL)
  KV_lds = KV_raw.reshape(BN, D_STR)
  S_part = UOp.alloc((4, BM, BN + 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  P_raw = UOp.alloc((BM * P_STR,), dtypes.half, addrspace=AddrSpace.LOCAL)
  P_lds = P_raw.reshape(BM, P_STR)
  m_lds = UOp.alloc((BM,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  l_lds = UOp.alloc((BM,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  a_lds = UOp.alloc((BM,), dtypes.float32, addrspace=AddrSpace.LOCAL)

  init_ml = UOp.group(
    m_lds[lane.minimum(15).valid(wid.eq(0) & (lane < 16))].store(UOp.const(-1e30, dtypes.float32)),
    l_lds[lane.minimum(15).valid(wid.eq(0) & (lane < 16))].store(UOp.const(0.0, dtypes.float32)),
  )

  # Preload this warp's 64-dim slice of Q [16, 64] into registers
  q_reg = UOp.alloc((2, 8, 2), dtypes.half, addrspace=AddrSpace.REG)
  q_st = []
  for im in range(2):
    ql = im * 8 + m_lane
    qpos = (q_start + ql).minimum(L - 1)
    qraw = rows_idx[row_base + qpos].load()
    qtok = ((qraw - S_p) if S_p else qraw).maximum(0).minimum(N - 1)
    vq = ql < q_len
    for ik in range(8):
      d0 = wid * 64 + ik * 8 + n_base
      q_st.append(q_reg[im, ik, 0].store(vq.where(q_in[qtok, h, d0].load(), UOp.const(0.0, dtypes.half))))
      q_st.append(q_reg[im, ik, 1].store(vq.where(q_in[qtok, h, d0 + 1].load(), UOp.const(0.0, dtypes.half))))
  q_reg = q_reg.after(*q_st)

  o_acc = UOp.alloc((2, 8, 2), dtypes.float32, addrspace=AddrSpace.REG)
  o_acc = o_acc.after(*[o_acc[im, id8, e].store(0.0) for im in range(2) for id8 in range(8) for e in range(2)])

  b_init = init_ml.barrier()
  m_lds, l_lds = m_lds.after(b_init), l_lds.after(b_init)

  n_kt = (L + BN - 1) // BN
  kt = UOp.range(n_kt, 100)
  k_base = kt * BN
  tile_active = k_base < (q_start + q_len)

  # 1. Cooperatively load K[k_base : k_base+16, 0:256] -> KV_lds (directly from k_pref when ktok < S_p)
  li = UOp.range(BN, 10)
  kpos = k_base + li
  vk = tile_active & (kpos < (q_start + q_len))
  ktok = rows_idx[(row_base + kpos.minimum(L - 1)).valid(vk)].load()
  KV_k_cur = KV_lds.after(kt)
  if S_p > 0:
    is_pk = ktok < S_p
    kp_i, kb_i = ktok.maximum(0).minimum(S_p - 1), (ktok - S_p).maximum(0).minimum(N - 1)
    k_v0 = is_pk.where(k_pref[kp_i.valid(vk & is_pk), kh, tid * 2].load(),
                       k_in[kb_i.valid(vk & is_pk.logical_not()), kh, tid * 2].load())
    k_v1 = is_pk.where(k_pref[kp_i.valid(vk & is_pk), kh, tid * 2 + 1].load(),
                       k_in[kb_i.valid(vk & is_pk.logical_not()), kh, tid * 2 + 1].load())
  else:
    ktok_s = ktok.maximum(0).minimum(N - 1)
    k_v0 = k_in[ktok_s.valid(vk), kh, tid * 2].load()
    k_v1 = k_in[ktok_s.valid(vk), kh, tid * 2 + 1].load()
  st_k = UOp.group(KV_k_cur[li, tid * 2].store(k_v0), KV_k_cur[li, tid * 2 + 1].store(k_v1)).end(li).barrier()
  K_r = KV_lds.after(st_k)

  # 2. Each warp computes its 64-dim partial of S[16, 16] via UOp.wmma
  s_acc = [[UOp.const((0.0, 0.0), dtypes.float32) for _ in range(2)] for _ in range(2)]
  for ik in range(8):
    d0 = wid * 64 + ik * 8
    kf_b = [UOp.stack(K_r[jn * 8 + n_base, d0 + m_lane].load(), K_r[jn * 8 + n_base + 1, d0 + m_lane].load()) for jn in range(2)]
    for im in range(2):
      qf_a = UOp.stack(q_reg[im, ik, 0].load(), q_reg[im, ik, 1].load())
      for jn in range(2):
        s_acc[im][jn] = UOp.wmma(qf_a, kf_b[jn], s_acc[im][jn], (8, 8, 8), 32)

  st_sp = [S_part.after(kt)[wid, im * 8 + m_lane, jn * 8 + n_base + e].store(s_acc[im][jn][e])
           for im in range(2) for jn in range(2) for e in range(2)]
  b_sp = UOp.group(*st_sp).barrier()
  Sp_r = S_part.after(b_sp)

  # 3. Online softmax: 4 warps x 2 half-warps = 8 rows simultaneously (2 steps for 16 rows)
  half_w, col16 = lane // 16, lane % 16
  k_pos_c = k_base + col16
  m_prev_lds, l_prev_lds = m_lds.after(kt), l_lds.after(kt)
  st_sm = []
  for r_step in range(2):
    r_local = wid * 4 + r_step * 2 + half_w
    q_pos_r = q_start + r_local
    causal = tile_active & (k_pos_c <= q_pos_r) & (r_local < q_len)
    raw_s = (Sp_r[0, r_local, col16].load() + Sp_r[1, r_local, col16].load() +
             Sp_r[2, r_local, col16].load() + Sp_r[3, r_local, col16].load()) * scale
    sv = causal.where(raw_s, UOp.const(-1e30, dtypes.float32))
    mx = sv
    for off in (8, 4, 2, 1):
      mx = mx.maximum(UOp.native(f"simd_shuffle_xor({{0}}, {off})", shape=(), dtype=dtypes.float32, fmt=True).call(mx))
    m_old = m_prev_lds[r_local].load()
    m_new = m_old.maximum(mx)
    alpha = tile_active.where(((m_old - m_new) * LOG2E).exp2(), UOp.const(1.0, dtypes.float32))
    pv = causal.where(((sv - m_new) * LOG2E).exp2(), UOp.const(0.0, dtypes.float32))
    ps = pv
    for off in (8, 4, 2, 1):
      ps = ps + UOp.native(f"simd_shuffle_xor({{0}}, {off})", shape=(), dtype=dtypes.float32, fmt=True).call(ps)
    l_new = l_prev_lds[r_local].load() * alpha + ps
    st_sm.append(P_lds.after(kt)[r_local, col16].store(pv.cast(dtypes.half)))
    st_sm.append(m_lds[r_local.valid(col16.eq(0))].store(m_new))
    st_sm.append(l_lds[r_local.valid(col16.eq(0))].store(l_new))
    st_sm.append(a_lds.after(kt)[r_local.valid(col16.eq(0))].store(alpha))

  # 4. Load V[k_base : k_base+16, 0:256] -> KV_lds (overwriting K) in parallel with softmax before one PV barrier
  li_v = UOp.range(BN, 11)
  kpos_v = k_base + li_v
  vv = tile_active & (kpos_v < (q_start + q_len))
  vtok = rows_idx[(row_base + kpos_v.minimum(L - 1)).valid(vv)].load()
  KV_v_cur = KV_lds.after(b_sp)
  v_off_kh = v_off + kh * hd
  if S_p > 0:
    is_pv = vtok < S_p
    vp_i, vb_i = vtok.maximum(0).minimum(S_p - 1), (vtok - S_p).maximum(0).minimum(N - 1)
    v_v0 = is_pv.where(v_pref[vp_i.valid(vv & is_pv), kh * hd + tid * 2].load(),
                       p_in[vb_i.valid(vv & is_pv.logical_not()), v_off_kh + tid * 2].load())
    v_v1 = is_pv.where(v_pref[vp_i.valid(vv & is_pv), kh * hd + tid * 2 + 1].load(),
                       p_in[vb_i.valid(vv & is_pv.logical_not()), v_off_kh + tid * 2 + 1].load())
  else:
    vtok_s = vtok.maximum(0).minimum(N - 1)
    v_v0 = p_in[vtok_s.valid(vv), v_off_kh + tid * 2].load()
    v_v1 = p_in[vtok_s.valid(vv), v_off_kh + tid * 2 + 1].load()
  st_v = UOp.group(KV_v_cur[li_v, tid * 2].store(v_v0), KV_v_cur[li_v, tid * 2 + 1].store(v_v1)).end(li_v)

  b_pv = UOp.group(*st_sm, st_v).barrier()
  P_raw_r, V_raw_r, a_r = P_raw.after(b_pv), KV_raw.after(b_pv), a_lds.after(b_pv)

  # 5. Rescale O_acc and accumulate O_acc += P[16, 16] @ V[16, wid*64 : +64] via UOp.wmma (half2 LDS reads)
  o_prev = o_acc.after(kt)
  oc = [[None] * 8 for _ in range(2)]
  for im in range(2):
    al = a_r[im * 8 + m_lane].load()
    for id8 in range(8):
      oc[im][id8] = UOp.stack(o_prev[im, id8, 0].load() * al, o_prev[im, id8, 1].load() * al)

  for ik in range(2):
    k0 = ik * 8
    pf_a = [_load2(P_raw_r, (im * 8 + m_lane) * P_STR + k0 + n_base) for im in range(2)]
    vf_b = [_load2(V_raw_r, (k0 + m_lane) * D_STR + wid * 64 + id8 * 8 + n_base) for id8 in range(8)]
    for im in range(2):
      for id8 in range(8):
        oc[im][id8] = UOp.wmma(pf_a[im], vf_b[id8], oc[im][id8], (8, 8, 8), 32)

  upd_o = UOp.group(*[o_acc[im, id8, e].store(oc[im][id8][e]) for im in range(2) for id8 in range(8) for e in range(2)]).barrier().end(kt)

  # 6. Normalize by l_lds and fuse * sigmoid(gate) on packed output store
  o_f, l_f = o_acc.after(upd_o), l_lds.after(upd_o)
  st_out = []
  for im in range(2):
    ql = im * 8 + m_lane
    vq = ql < q_len
    qpos = (q_start + ql).minimum(L - 1)
    qraw = rows_idx[row_base + qpos].load()
    qtok = ((qraw - S_p) if S_p else qraw).maximum(0).minimum(N - 1)
    inv_l = l_f[ql].load().reciprocal()
    for id8 in range(8):
      for e in range(2):
        d = wid * 64 + id8 * 8 + n_base + e
        val = o_f[im, id8, e].load() * inv_l
        if gated:
          gv = p_in[qtok, h * 2 * hd + hd + d].load().float()
          val = val * ((-gv * LOG2E).exp2() + 1.0).reciprocal()
        st_out.append(out[qtok.valid(vq), h * hd + d].store(val.cast(dtypes.half)))
  sfx = f"_p{S_p}" if S_p > 0 else ""
  return UOp.group(*st_out).end(lane, wid, th).sink(arg=KernelInfo(name=f"packed_flash_attn_wmma_{N}_{R}_{L}_{NT}{sfx}", opts_to_apply=()))

def _build_ro_qk_norm_rope(M_ro:int, N:int, nh:int, nkv:int, hd:int, rope_dim:int,
                           gated:bool, has_norm:bool, eps:float, max_pos:int) -> UOp:
  """Readout-only QK-RMSNorm + partial NeoX RoPE: rotates `q_ro [M_ro, Q_P]` and `kv_all [N, 2*nkv*hd]` (K half)."""
  assert hd % 32 == 0 and rope_dim == 64 and hd >= 64
  E = hd // 32
  q_stride = 2 * hd if gated else hd
  q_out = UOp.param(0, dtypes.half, (M_ro, nh, hd))
  k_out = UOp.param(1, dtypes.half, (N, nkv, hd))
  q_ro = UOp.param(2, dtypes.half, (M_ro, nh * q_stride))
  kv_all = UOp.param(3, dtypes.half, (N, 2 * nkv * hd))
  qw = UOp.param(4, dtypes.half, (hd,))
  kw = UOp.param(5, dtypes.half, (hd,))
  freqs = UOp.param(6, dtypes.float32, (max_pos, rope_dim))
  ro_pos = UOp.param(7, dtypes.int32, (M_ro,))
  tok_pos = UOp.param(8, dtypes.int32, (N,))

  n_qw = M_ro * nh
  w = UOp.range(n_qw + N * nkv, 0, AxisType.GLOBAL)
  is_q = w < n_qw
  qi = (w // nh).minimum(M_ro - 1)
  hq = w % nh
  kw_i = (w - n_qw).maximum(0)
  ki = (kw_i // nkv).minimum(N - 1)
  hk = kw_i % nkv
  lane = UOp.range(32, -1, AxisType.WARP)

  reg = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  c_r = UOp.range(E, 2, AxisType.UPCAST)
  d = lane + c_r * 32
  xv = is_q.where(q_ro[qi, hq * q_stride + d].load().float(), kv_all[ki, hk * hd + d].load().float())
  reg = reg.after(reg[c_r].store(xv).end(c_r))

  if has_norm:
    sq = _simd_sum((reg[c_r].load() * reg[c_r].load()).reduce(c_r, arg=(Ops.ADD, 0)))
    inv = (sq * (1.0 / hd) + eps).rsqrt()
    vals = []
    for c in range(E):
      dc = lane + c * 32
      wt = is_q.where(qw[dc].load(), kw[dc].load())
      vals.append(((reg[c].load() * inv).cast(dtypes.half) * wt).float())
  else:
    vals = [reg[c].load() for c in range(E)]

  pos = is_q.where(ro_pos[qi].load(), tok_pos[ki].load()).maximum(0).minimum(max_pos - 1)
  cos, sin = freqs[pos, lane].load(), freqs[pos, lane + 32].load()
  out_vals = [vals[0] * cos - vals[1] * sin, vals[1] * cos + vals[0] * sin] + vals[2:]

  sts = []
  for c in range(E):
    dc = lane + c * 32
    vh = out_vals[c].cast(dtypes.half)
    sts.append(q_out[qi.valid(is_q), hq, dc].store(vh))
    sts.append(k_out[ki.valid(is_q.eq(False)), hk, dc].store(vh))
  return UOp.group(*sts).end(lane, w).sink(arg=KernelInfo(name=f"ro_qk_norm_rope_{M_ro}_{N}", opts_to_apply=()))

def _build_ro_flash_attn(M_ro:int, N:int, R:int, L:int, nh:int, nkv:int, hd:int, gated:bool, S_p:int=0) -> UOp:
  """Zero-LDS readout-only packed Flash Attention: runs only on `M_ro` query tokens, with optional zero-copy `S_p` KV prefix."""
  assert hd % 32 == 0 and nh % nkv == 0
  E, gqa, scale = hd // 32, nh // nkv, hd ** -0.5
  q_stride = 2 * hd if gated else hd

  out = UOp.param(0, dtypes.half, (M_ro, nh * hd))
  q_in = UOp.param(1, dtypes.half, (M_ro, nh, hd))
  k_in = UOp.param(2, dtypes.half, (N, nkv, hd))
  kv_all = UOp.param(3, dtypes.half, (N, 2 * nkv * hd))
  q_ro = UOp.param(4, dtypes.half, (M_ro, nh * q_stride))
  rows_idx = UOp.param(5, dtypes.int32, (R * L,))
  ro_row = UOp.param(6, dtypes.int32, (M_ro,))
  ro_pos = UOp.param(7, dtypes.int32, (M_ro,))
  k_pref = UOp.param(8, dtypes.half, (S_p, nkv, hd)) if S_p > 0 else None
  v_pref = UOp.param(9, dtypes.half, (S_p, nkv * hd)) if S_p > 0 else None

  th = UOp.range(M_ro * nh, 0, AxisType.GLOBAL)
  t, h = th // nh, th % nh
  kh = h // gqa
  lane = UOp.range(32, -1, AxisType.WARP)

  q_reg = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  e_u = UOp.range(E, 2, AxisType.UPCAST)
  d = lane * E + e_u
  q_reg = q_reg.after(q_reg[e_u].store(q_in[t, h, d].load().float() * scale).end(e_u))

  acc = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  m_reg = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  l_reg = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[e_u].store(0.0).end(e_u))
  m_reg = m_reg.after(m_reg[0].store(UOp.const(-1e30, dtypes.float32)))
  l_reg = l_reg.after(l_reg[0].store(0.0))

  row_base = ro_row[t].load().maximum(0).minimum(R - 1) * L
  pos_t = ro_pos[t].load().maximum(0).minimum(L - 1)
  v_off_kh = nkv * hd + kh * hd

  if S_p > 0:
    # Phase 1: static prefix 0..S_p-1 directly from k_pref/v_pref (no rows_idx indirection, no branch)
    jp = UOp.range(S_p, 10)
    sc_p = _simd_sum((q_reg[e_u].load() * k_pref[jp, kh, d].load().float()).reduce(e_u, arg=(Ops.ADD, 0)))
    mp, lp = m_reg.after(jp)[0].load(), l_reg.after(jp)[0].load()
    mn_p = mp.maximum(sc_p)
    al_p = ((mp - mn_p) * LOG2E).exp2()
    w_p = ((sc_p - mn_p) * LOG2E).exp2()
    upd1 = UOp.group(
      m_reg[0].store(mn_p),
      l_reg[0].store(lp * al_p + w_p),
      acc[e_u].store(acc.after(jp)[e_u].load() * al_p + w_p * v_pref[jp, kh * hd + d].load().float()).end(e_u),
    ).end(jp)
    acc, m_reg, l_reg = acc.after(upd1), m_reg.after(upd1), l_reg.after(upd1)

    # Phase 2: branch tokens S_p..L-1 directly from k_in/kv_all (no is_p branch)
    jb = UOp.range(L - S_p, 11)
    valid_j = (S_p + jb) <= pos_t
    kj = (rows_idx[(row_base + S_p + jb).valid(valid_j)].load() - S_p).maximum(0).minimum(N - 1)
    dot_b = (q_reg[e_u].load() * k_in[kj.valid(valid_j), kh, d].load().float()).reduce(e_u, arg=(Ops.ADD, 0))
    sc_b = valid_j.where(_simd_sum(dot_b), UOp.const(-1e30, dtypes.float32))
    mb, lb = m_reg.after(jb)[0].load(), l_reg.after(jb)[0].load()
    mn_b = mb.maximum(sc_b)
    al_b = ((mb - mn_b) * LOG2E).exp2()
    w_b = valid_j.where(((sc_b - mn_b) * LOG2E).exp2(), UOp.const(0.0, dtypes.float32))
    upd = UOp.group(
      m_reg[0].store(mn_b),
      l_reg[0].store(lb * al_b + w_b),
      acc[e_u].store(acc.after(jb)[e_u].load() * al_b + w_b * kv_all[kj.valid(valid_j), v_off_kh + d].load().float()).end(e_u),
    ).end(jb)
  else:
    j = UOp.range(L, 10)
    valid_j = j <= pos_t
    kj_s = rows_idx[(row_base + j).valid(valid_j)].load().maximum(0).minimum(N - 1)
    dot_l = (q_reg[e_u].load() * k_in[kj_s.valid(valid_j), kh, d].load().float()).reduce(e_u, arg=(Ops.ADD, 0))
    score = valid_j.where(_simd_sum(dot_l), UOp.const(-1e30, dtypes.float32))
    m_prev, l_prev = m_reg.after(j)[0].load(), l_reg.after(j)[0].load()
    m_new = m_prev.maximum(score)
    alpha = ((m_prev - m_new) * LOG2E).exp2()
    w = valid_j.where(((score - m_new) * LOG2E).exp2(), UOp.const(0.0, dtypes.float32))
    upd = UOp.group(
      m_reg[0].store(m_new),
      l_reg[0].store(l_prev * alpha + w),
      acc[e_u].store(acc.after(j)[e_u].load() * alpha + w * kv_all[kj_s.valid(valid_j), v_off_kh + d].load().float()).end(e_u),
    ).end(j)

  inv_l = l_reg.after(upd)[0].load().reciprocal()
  acc_f = acc.after(upd)
  e_o = UOp.range(E, 3, AxisType.UPCAST)
  d_o = lane * E + e_o
  val = acc_f[e_o].load() * inv_l
  if gated:
    gv = q_ro[t, h * 2 * hd + hd + d_o].load().float()
    val = val * ((-gv * LOG2E).exp2() + 1.0).reciprocal()
  sfx = f"_p{S_p}" if S_p > 0 else ""
  return out[t, h * hd + d_o].store(val.cast(dtypes.half)).end(e_o).end(lane, th).sink(
    arg=KernelInfo(name=f"ro_flash_attn_{M_ro}_{N}_{L}{sfx}", opts_to_apply=()))

@functools.cache
def get_ro_qk_norm_rope(M_ro:int, N:int, nh:int, nkv:int, hd:int, rope_dim:int,
                        gated:bool, has_norm:bool, eps:float, max_pos:int) -> UOp:
  return _emit(f"ro_qk_norm_rope_{M_ro}_{N}_{nh}_{nkv}_{hd}_{int(gated)}_{int(has_norm)}",
               _build_ro_qk_norm_rope(M_ro, N, nh, nkv, hd, rope_dim, gated, has_norm, eps, max_pos))

@functools.cache
def get_ro_flash_attn(M_ro:int, N:int, R:int, L:int, nh:int, nkv:int, hd:int, gated:bool, S_p:int=0) -> UOp:
  sfx = f"_p{S_p}" if S_p > 0 else ""
  return _emit(f"ro_flash_attn_{M_ro}_{N}_{R}_{L}_{nh}_{nkv}_{hd}_{int(gated)}{sfx}",
               _build_ro_flash_attn(M_ro, N, R, L, nh, nkv, hd, gated, S_p))

def packed_attention_readout(blk, q_ro:Tensor, kv_all:Tensor, pk,
                             k_prefix:Tensor|None=None, v_prefix:Tensor|None=None) -> Tensor|None:
  """Readout-only packed Flash Attention: `q_ro (M_ro, Q_P)` + `kv_all (N, 2*nkv*hd)` -> `(M_ro, nh*hd)` f16."""
  cfg = blk.config
  nh, nkv, hd, rope_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim
  if not enabled() or not _is_metal(q_ro.device) or q_ro.dtype is not dtypes.float16 or kv_all.dtype is not dtypes.float16:
    return None
  if rope_dim != 64 or hd % 32 != 0 or hd < 64 or nh % nkv != 0: return None
  has_norm = (cfg.qk_norm == hd)
  if cfg.qk_norm and not has_norm: return None
  M_ro, N = q_ro.shape[-2], pk.N
  freqs = blk.freqs_cis
  if freqs.dtype is not dtypes.float32: freqs = freqs.float()
  max_pos = freqs.shape[0]
  qw = blk.attn_q_norm.weight if has_norm else Tensor.ones(hd, dtype=dtypes.half, device=q_ro.device)
  kw = blk.attn_k_norm.weight if has_norm else qw
  if qw.dtype is not dtypes.float16 or kw.dtype is not dtypes.float16: return None

  k1 = get_ro_qk_norm_rope(M_ro, N, nh, nkv, hd, rope_dim, cfg.attn_output_gate, has_norm, cfg.norm_eps, max_pos)
  dev = q_ro.device
  q2, kv2 = q_ro.reshape(M_ro, -1).contiguous().uop, kv_all.reshape(N, -1).contiguous().uop
  q_buf = UOp.alloc((M_ro, nh, hd), dtypes.half, device=dev)
  k_buf = UOp.alloc((N, nkv, hd), dtypes.half, device=dev)
  c1 = k1.call(q_buf, k_buf, q2, kv2, qw.contiguous().uop, kw.contiguous().uop,
               freqs.contiguous().uop, pk.ro_pos.uop, pk.tok_pos.uop)
  out = UOp.alloc((M_ro, nh * hd), dtypes.half, device=dev)
  if k_prefix is not None and v_prefix is not None:
    S_p = k_prefix.shape[0]
    k2 = get_ro_flash_attn(M_ro, N, pk.R, pk.L, nh, nkv, hd, cfg.attn_output_gate, S_p)
    c2 = k2.call(out, q_buf.after(c1), k_buf.after(c1), kv2, q2, pk.rows_idx.uop, pk.ro_row.uop, pk.ro_pos.uop,
                 k_prefix.contiguous().uop, v_prefix.contiguous().uop)
  else:
    k2 = get_ro_flash_attn(M_ro, N, pk.R, pk.L, nh, nkv, hd, cfg.attn_output_gate)
    c2 = k2.call(out, q_buf.after(c1), k_buf.after(c1), kv2, q2, pk.rows_idx.uop, pk.ro_row.uop, pk.ro_pos.uop)
  return Tensor(out.after(c2))

@functools.cache
def get_qk_norm_rope(N:int, nh:int, nkv:int, hd:int, rope_dim:int, P:int, k_off:int,
                     gated:bool, has_norm:bool, eps:float, max_pos:int) -> UOp:
  return _emit(f"qk_norm_rope_{N}_{nh}_{nkv}_{hd}_{P}_{int(gated)}_{int(has_norm)}",
               _build_qk_norm_rope(N, nh, nkv, hd, rope_dim, P, k_off, gated, has_norm, eps, max_pos))

@functools.cache
def get_packed_flash_attn(N:int, R:int, L:int, nh:int, nkv:int, hd:int, P:int, v_off:int, gated:bool) -> UOp:
  return _emit(f"flash_attn_{N}_{R}_{L}_{nh}_{nkv}_{hd}_{P}_{int(gated)}",
               _build_packed_flash_attn(N, R, L, nh, nkv, hd, P, v_off, gated))

@functools.cache
def get_packed_flash_attn_wmma(N:int, R:int, L:int, NT:int, nh:int, nkv:int, hd:int, P:int, v_off:int,
                               gated:bool, S_p:int=0) -> UOp:
  sfx = f"_p{S_p}" if S_p > 0 else ""
  return _emit(f"flash_attn_wmma_{N}_{R}_{L}_{NT}_{nh}_{nkv}_{hd}_{P}_{int(gated)}{sfx}",
               _build_packed_flash_attn_wmma(N, R, L, NT, nh, nkv, hd, P, v_off, gated, S_p))

def packed_attention(blk, p:Tensor, pk, k_prefix:Tensor|None=None, v_prefix:Tensor|None=None,
                     return_kv:int=0) -> Tensor|tuple[Tensor, Tensor, Tensor]|None:
  """Run fused QK-norm+RoPE and packed Flash Attention directly on fused `p = _qkv(x)` -> `(N, nh*hd)` f16.

  When `return_kv > 0`, returns `(out, k_rot[:return_kv], v[:return_kv])` for prefix caching."""
  cfg = blk.config
  nh, nkv, hd, rope_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim
  if not enabled() or not _is_metal(p.device) or p.dtype is not dtypes.float16: return None
  if rope_dim != 64 or hd % 32 != 0 or hd < 64 or nh % nkv != 0: return None
  has_norm = (cfg.qk_norm == hd)
  if cfg.qk_norm and not has_norm: return None
  N, P = pk.N, p.shape[-1]
  k_off, v_off = blk._qkv_ends
  freqs = blk.freqs_cis
  if freqs.dtype is not dtypes.float32: freqs = freqs.float()
  max_pos = freqs.shape[0]
  qw = blk.attn_q_norm.weight if has_norm else Tensor.ones(hd, dtype=dtypes.half, device=p.device)
  kw = blk.attn_k_norm.weight if has_norm else qw
  if qw.dtype is not dtypes.float16 or kw.dtype is not dtypes.float16: return None

  k1 = get_qk_norm_rope(N, nh, nkv, hd, rope_dim, P, k_off, cfg.attn_output_gate, has_norm, cfg.norm_eps, max_pos)
  dev = p.device
  p_2d = p.reshape(N, P).contiguous()
  p2 = p_2d.uop
  q_buf = UOp.alloc((N, nh, hd), dtypes.half, device=dev)
  k_buf = UOp.alloc((N, nkv, hd), dtypes.half, device=dev)
  c1 = k1.call(q_buf, k_buf, p2, qw.contiguous().uop, kw.contiguous().uop, freqs.contiguous().uop, pk.tok_pos.uop)
  out = UOp.alloc((N, nh * hd), dtypes.half, device=dev)
  if k_prefix is not None and v_prefix is not None:
    if hd != 256 or pk.q_tiles is None: return None
    S_p = k_prefix.shape[0]
    k2 = get_packed_flash_attn_wmma(N, pk.R, pk.L, pk.q_tiles.shape[0], nh, nkv, hd, P, v_off, cfg.attn_output_gate, S_p)
    c2 = k2.call(out, q_buf.after(c1), k_buf.after(c1), p2, pk.rows_idx.uop, pk.q_tiles.uop,
                 k_prefix.contiguous().uop, v_prefix.contiguous().uop)
    return Tensor(out.after(c2))
  if hd == 256 and pk.q_tiles is not None:
    k2 = get_packed_flash_attn_wmma(N, pk.R, pk.L, pk.q_tiles.shape[0], nh, nkv, hd, P, v_off, cfg.attn_output_gate)
    c2 = k2.call(out, q_buf.after(c1), k_buf.after(c1), p2, pk.rows_idx.uop, pk.q_tiles.uop)
  else:
    k2 = get_packed_flash_attn(N, pk.R, pk.L, nh, nkv, hd, P, v_off, cfg.attn_output_gate)
    c2 = k2.call(out, q_buf.after(c1), k_buf.after(c1), p2, pk.rows_idx.uop, pk.tok_row.uop, pk.tok_pos.uop)
  res = Tensor(out.after(c2))
  if return_kv > 0:
    return res, Tensor(k_buf.after(c1))[:return_kv].contiguous(), p_2d[:return_kv, v_off:].contiguous()
  return res
