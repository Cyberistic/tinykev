"""Hand-scheduled and threadgroup-tiled Metal WMMA GEMMs in the `slopkernels` style.

Two paths for `float16` `Linear` layers:
* Pre-transposed weights `W_T [K, N]` (`matmul_t`, `swiglu_matmul_t`): both `A [M, K]` and `W_T [K, N]` load
  coalesced along the inner axis. Large-K / wide-N shapes use a pure UIR threadgroup-tiled WMMA kernel
  (`AddrSpace.LOCAL` + coalesced `uint4` loads + unrolled `UOp.wmma`); other shapes use a coalesced no-LDS
  WMMA schedule (`OptOps.TC` + explicit tile splits).
* Standard weights `W [N, K]` (`matmul`): kept as fallback and for unit tests.

All kernels are rendered to `tinykev/uir/*.uir` and parsed back via `parse_uir`.
"""
from __future__ import annotations

import functools, math, os, pathlib, threading

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.uir import parse_uir, render_uir

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
LOG2E = math.log2(math.e)

# ---- legacy [N, K] schedule (fallback / tests) ----
TILES = {(1024, 6144): (4, 4, 4), (1024, 3584): (4, 2, 4), (3584, 1024): (2, 2, 4), (1024, 2048): (4, 2, 8),
         (2048, 1024): (4, 2, 4), (1024, 512): (3, 2, 8), (1024, 4096): (4, 4, 2), (1024, 16): (4, 2, 1)}
TILES_M = {(112, 1024, 6144): (7, 4, 8), (112, 1024, 2048): (2, 2, 2), (112, 2048, 1024): (2, 2, 2),
           (112, 1024, 3584): (2, 2, 2), (112, 3584, 1024): (2, 2, 8), (112, 1024, 4096): (7, 2, 2),
           (112, 1024, 8224): (7, 2, 1)}

# ---- pre-transposed [K, N] plans (measured on Apple Silicon Metal) ----
# Pure UIR threadgroup-tiled WMMA: (BM, BN, BK, WM, WN, pad)
TKN_SHAPES: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {
  (1024, 8224): (16, 32, 32, 1, 2, 4),   # DeltaNet fused qkv|gate|beta|alpha
  (1024, 5120): (16, 32, 32, 1, 2, 4),   # Attention fused q|k|v
  (3584, 1024): (16, 32, 32, 1, 2, 4),   # MLP down_proj
  (2048, 1024): (16, 32, 32, 1, 2, 4),   # ssm_out / attn_output
  (1024, 7168): (16, 64, 32, 1, 2, 4),   # fused gate+up
}
# Split-K shapes whose partials `(SK, M, N)` reduce inside the immediately-following `add_rmsnorm`:
# (BM, BN, BK, WM, WN, pad, SK)
TKN_SPLITK: dict[tuple[int, int], tuple[int, int, int, int, int, int, int]] = {
  (2048, 1024): (16, 32, 32, 1, 2, 4, 4),   # ssm_out / attn_output
  (3584, 1024): (16, 64, 32, 1, 2, 4, 4),   # MLP down_proj
}
# Coalesced no-LDS WMMA on [K, N]: (mu, nu, nl)
KN_TILES: dict[tuple[int, int], tuple[int, int, int]] = {
  (2048, 1024): (2, 2, 2),               # ssm_out / attn_output
  (1024, 5120): (2, 4, 2),               # Attention fused q|k|v
  (1024, 3584): (2, 4, 2),               # MLP gate / up
  (1024, 4096): (2, 4, 2),
  (1024, 6144): (2, 4, 2),
  (1024, 2048): (2, 2, 4),
}

def _load2(buf_flat:UOp, off:UOp) -> UOp:
  return UOp(Ops.SHRINK, src=(buf_flat, off, UOp.const(2))).load()

def _emit_uir(key:str, u:UOp, rewrite:bool=False) -> UOp:
  with Context(SPEC=0):
    if rewrite: u = graph_rewrite(u, symbolic_simple, enter_calls=True)
    uir = render_uir(u, color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"{key}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def _plan(M:int, K:int, N:int) -> tuple[int, int, int]|None:
  base = -(-M // 8)
  fits = lambda t: t[0] <= base // 2 and base % t[0] == 0 and N % (8 * t[1] * t[2]) == 0
  if (t := TILES_M.get((M, K, N))) is not None and fits(t): return t
  mu, nu, nl = TILES.get((K, N), (4, 4, 4))
  mu = next((m for m in (mu, 4, 3, 2) if m <= base // 2 and base % m == 0), None)
  if mu is None: return None
  if N % (8 * nu * nl):
    nu, nl = next(((a, b) for a, b in ((4, 4), (2, 4), (4, 2), (2, 2), (2, 8), (8, 2)) if N % (8 * a * b) == 0), (0, 0))
  return None if nu == 0 else (mu, nu, nl)

def _plan_kn(M:int, K:int, N:int) -> tuple[int, int, int]|None:
  base = -(-M // 8)
  mu, nu, nl = KN_TILES.get((K, N), (2, 4, 2))
  mu = next((m for m in (mu, 4, 3, 2) if m <= base // 2 and base % m == 0), None)
  if mu is None: return None
  if N % (8 * nu * nl):
    nu, nl = next(((a, b) for a, b in ((4, 2), (2, 4), (4, 4), (2, 2), (2, 1)) if N % (8 * a * b) == 0), (0, 0))
  return None if nu == 0 else (mu, nu, nl)

def _opts(mu:int, nu:int, nl:int) -> tuple[Opt, ...]:
  base = (Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.SPLIT, 0, (mu, AxisType.UPCAST)),
          Opt(OptOps.SPLIT, 1, (nu, AxisType.UPCAST)))
  return base + ((Opt(OptOps.SPLIT, 1, (nl, AxisType.LOCAL)),) if nl > 1 else ())

def _build(M:int, K:int, N:int) -> UOp:
  out, a, w = (UOp.param(0, dtypes.float16, (M, N)), UOp.param(1, dtypes.float16, (M, K)),
               UOp.param(2, dtypes.float16, (N, K)))
  i, j, k = UOp.range(M, 0, AxisType.WEAK), UOp.range(N, 1, AxisType.WEAK), UOp.range(K, 2, AxisType.WEAK)
  acc = (a[i, k].load() * w[j, k].load()).reduce(k, arg=(Ops.ADD, 0))
  mu, nu, nl = _plan(M, K, N)
  return out[i, j].store(acc).end(i, j).sink(arg=KernelInfo(opts_to_apply=_opts(mu, nu, nl)))

def _build_kn(M:int, K:int, N:int) -> UOp:
  out, a, wt = (UOp.param(0, dtypes.float16, (M, N)), UOp.param(1, dtypes.float16, (M, K)),
                UOp.param(2, dtypes.float16, (K, N)))
  i, j, k = UOp.range(M, 0, AxisType.WEAK), UOp.range(N, 1, AxisType.WEAK), UOp.range(K, 2, AxisType.WEAK)
  acc = (a[i, k].load() * wt[k, j].load()).reduce(k, arg=(Ops.ADD, 0))
  mu, nu, nl = _plan_kn(M, K, N)
  return out[i, j].store(acc).end(i, j).sink(arg=KernelInfo(opts_to_apply=_opts(mu, nu, nl)))

def _build_tkn(M:int, K:int, N:int, BM:int, BN:int, BK:int, WM:int, WN:int, pad:int=4, sk:int=1) -> UOp:
  """Pure Metal UIR threadgroup-tiled WMMA GEMM on pre-transposed W_T[K, N] (`half2` LDS reads, f32 acc, optional Split-K)."""
  TM, TN, THREADS = (BM // WM) // 8, (BN // WN) // 8, WM * WN * 32
  K_PER = K // sk
  out = UOp.param(0, dtypes.half, (sk, M, N)) if sk > 1 else UOp.param(0, dtypes.half, (M, N))
  a, wt = UOp.param(1, dtypes.half, (M, K)), UOp.param(2, dtypes.half, (K, N))
  bn = UOp.range(N // BN, 0, AxisType.GLOBAL)
  skr = UOp.range(sk, 1, AxisType.GLOBAL) if sk > 1 else None
  bm = UOp.range(M // BM, 4, AxisType.GLOBAL)
  wm, wn, lane = UOp.range(WM, 2, AxisType.LOCAL), UOp.range(WN, 3, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  tid = (wm * WN + wn) * 32 + lane
  m_lane = ((lane >> 1) & 3) + ((lane >> 4) * 4)
  n_base = ((lane & 1) * 2) + (((lane >> 3) & 1) * 4)

  a_str, b_str = BK + pad, BN + pad
  A_raw = UOp.alloc((BM * a_str,), dtypes.half, addrspace=AddrSpace.LOCAL)
  B_raw = UOp.alloc((BK * b_str,), dtypes.half, addrspace=AddrSpace.LOCAL)

  acc = UOp.alloc((TM, TN, 2), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(*[acc[ti, tj, te].store(UOp.const(0.0, dtypes.float32)) for ti in range(TM) for tj in range(TN) for te in range(2)])

  kt = UOp.range(K_PER // BK, 100)
  k0 = (skr * K_PER + kt * BK) if skr is not None else (kt * BK)

  a_u32 = a.flatten().bitcast(dtypes.uint32).reshape(M, K // 2)
  A_u32 = A_raw.after(kt).bitcast(dtypes.uint32).reshape(BM, a_str // 2)
  a_vec = 4 if (BK // 2) % 4 == 0 and ((BM * BK // 2) // 4) % THREADS == 0 else (2 if (BK // 2) % 2 == 0 and ((BM * BK // 2) // 2) % THREADS == 0 else 1)
  ai, au = UOp.range((BM * BK // 2) // (THREADS * a_vec), 20), UOp.range(a_vec, 21, AxisType.UPCAST)
  a_idx = (ai * THREADS + tid) * a_vec + au
  a_st = A_u32[a_idx // (BK // 2), a_idx % (BK // 2)].store(a_u32[bm * BM + a_idx // (BK // 2), k0 // 2 + a_idx % (BK // 2)].load()).end(au, ai)

  wt_u32 = wt.flatten().bitcast(dtypes.uint32).reshape(K, N // 2)
  B_u32 = B_raw.after(kt).bitcast(dtypes.uint32).reshape(BK, b_str // 2)
  b_vec = 4 if (BN // 2) % 4 == 0 and ((BK * BN // 2) // 4) % THREADS == 0 else (2 if (BN // 2) % 2 == 0 and ((BK * BN // 2) // 2) % THREADS == 0 else 1)
  bi, bu = UOp.range((BK * BN // 2) // (THREADS * b_vec), 22), UOp.range(b_vec, 23, AxisType.UPCAST)
  b_idx = (bi * THREADS + tid) * b_vec + bu
  b_st = B_u32[b_idx // (BN // 2), b_idx % (BN // 2)].store(wt_u32[k0 + b_idx // (BN // 2), bn * (BN // 2) + b_idx % (BN // 2)].load()).end(bu, bi)

  A_r, B_r = A_raw.after(a_st, b_st), B_raw.after(a_st, b_st)
  cur = [[UOp.stack(acc.after(kt)[ti, tj, 0].load(), acc.after(kt)[ti, tj, 1].load()) for tj in range(TN)] for ti in range(TM)]
  for kk in range(BK // 8):
    ks = kk * 8
    af = [_load2(A_r, (wm * (TM * 8) + ti * 8 + m_lane) * a_str + ks + n_base) for ti in range(TM)]
    bf = [_load2(B_r, (ks + m_lane) * b_str + wn * (TN * 8) + tj * 8 + n_base) for tj in range(TN)]
    for ti in range(TM):
      for tj in range(TN):
        cur[ti][tj] = UOp.wmma(af[ti], bf[tj], cur[ti][tj], (8, 8, 8), 32)
  upd = UOp.group(*[acc[ti, tj, te].store(cur[ti][tj][te]) for ti in range(TM) for tj in range(TN) for te in range(2)]).barrier().end(kt)

  acc_f = acc.after(upd)
  sts = []
  for ti in range(TM):
    om = bm * BM + wm * (TM * 8) + ti * 8 + m_lane
    for tj in range(TN):
      on = bn * BN + wn * (TN * 8) + tj * 8 + n_base
      v0, v1 = acc_f[ti, tj, 0].load().cast(dtypes.half), acc_f[ti, tj, 1].load().cast(dtypes.half)
      if sk > 1: sts += [out[skr, om, on].store(v0), out[skr, om, on + 1].store(v1)]
      else: sts += [out[om, on].store(v0), out[om, on + 1].store(v1)]
  ends = [lane, wn, wm, bm] + ([skr] if skr is not None else []) + [bn]
  name = f"tkn_{BM}_{BN}_{BK}_{WM}_{WN}" + (f"_sk{sk}" if sk > 1 else "")
  return UOp.group(*sts).end(*ends).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def _build_tkn_swiglu(M:int, K:int, N:int, BM:int=16, BN:int=32, BK:int=16, WM:int=1, WN:int=2) -> UOp:
  """Zero-LDS, zero-barrier fused MLP gate+up+SwiGLU WMMA kernel (`half2` loads, f32 acc): `silu(a @ wgt) * (a @ wut)`."""
  TM, TN = (BM // WM) // 8, (BN // WN) // 8
  out = UOp.param(0, dtypes.half, (M, N))
  a, wgt, wut = UOp.param(1, dtypes.half, (M, K)), UOp.param(2, dtypes.half, (K, N)), UOp.param(3, dtypes.half, (K, N))
  bn, bm = UOp.range(N // BN, 0, AxisType.GLOBAL), UOp.range(M // BM, 1, AxisType.GLOBAL)
  wm, wn, lane = UOp.range(WM, 2, AxisType.LOCAL), UOp.range(WN, 3, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  m_lane = ((lane >> 1) & 3) + ((lane >> 4) * 4)
  n_base = ((lane & 1) * 2) + (((lane >> 3) & 1) * 4)

  ag = UOp.alloc((TM, TN, 2), dtypes.float32, addrspace=AddrSpace.REG)
  au_r = UOp.alloc((TM, TN, 2), dtypes.float32, addrspace=AddrSpace.REG)
  ag = ag.after(*[ag[i, j, e].store(UOp.const(0.0, dtypes.float32)) for i in range(TM) for j in range(TN) for e in range(2)])
  au_r = au_r.after(*[au_r[i, j, e].store(UOp.const(0.0, dtypes.float32)) for i in range(TM) for j in range(TN) for e in range(2)])

  kt = UOp.range(K // BK, 100)
  k0 = kt * BK
  a_flat, wg_flat, wu_flat = a.flatten(), wgt.flatten(), wut.flatten()
  cg = [[UOp.stack(ag.after(kt)[i, j, 0].load(), ag.after(kt)[i, j, 1].load()) for j in range(TN)] for i in range(TM)]
  cu = [[UOp.stack(au_r.after(kt)[i, j, 0].load(), au_r.after(kt)[i, j, 1].load()) for j in range(TN)] for i in range(TM)]
  for kk in range(BK // 8):
    ks = k0 + kk * 8
    af = [_load2(a_flat, (bm * BM + wm * (TM * 8) + i * 8 + m_lane) * K + ks + n_base) for i in range(TM)]
    bgf = [_load2(wg_flat, (ks + m_lane) * N + bn * BN + wn * (TN * 8) + j * 8 + n_base) for j in range(TN)]
    buf = [_load2(wu_flat, (ks + m_lane) * N + bn * BN + wn * (TN * 8) + j * 8 + n_base) for j in range(TN)]
    for i in range(TM):
      for j in range(TN):
        cg[i][j] = UOp.wmma(af[i], bgf[j], cg[i][j], (8, 8, 8), 32)
        cu[i][j] = UOp.wmma(af[i], buf[j], cu[i][j], (8, 8, 8), 32)
  upd = UOp.group(
    *[ag[i, j, e].store(cg[i][j][e]) for i in range(TM) for j in range(TN) for e in range(2)],
    *[au_r[i, j, e].store(cu[i][j][e]) for i in range(TM) for j in range(TN) for e in range(2)],
  ).end(kt)

  ag_f, au_f = ag.after(upd), au_r.after(upd)
  sts = []
  for i in range(TM):
    om = bm * BM + wm * (TM * 8) + i * 8 + m_lane
    for j in range(TN):
      on = bn * BN + wn * (TN * 8) + j * 8 + n_base
      for e in range(2):
        gv, uv = ag_f[i, j, e].load(), au_f[i, j, e].load()
        sts.append(out[om, on + e].store((gv * ((-gv * LOG2E).exp2() + 1.0).reciprocal() * uv).cast(dtypes.half)))
  return UOp.group(*sts).end(lane, wn, wm, bm, bn).sink(arg=KernelInfo(name=f"tkn_swiglu_{BM}_{BN}_{BK}", opts_to_apply=()))

@functools.cache
def get_gemm(M:int, K:int, N:int) -> UOp: return _emit_uir(f"gemm_{M}_{K}_{N}", _build(M, K, N))

@functools.cache
def get_gemm_kn(M:int, K:int, N:int) -> UOp: return _emit_uir(f"gemm_kn_{M}_{K}_{N}", _build_kn(M, K, N))

@functools.cache
def get_gemm_tkn(M:int, K:int, N:int, BM:int, BN:int, BK:int, WM:int, WN:int, pad:int, sk:int=1) -> UOp:
  sfx = f"_sk{sk}" if sk > 1 else ""
  return _emit_uir(f"gemm_tkn_{M}_{K}_{N}_{BM}_{BN}_{BK}{sfx}", _build_tkn(M, K, N, BM, BN, BK, WM, WN, pad, sk), rewrite=True)

@functools.cache
def get_swiglu_tkn(M:int, K:int, N:int) -> UOp:
  return _emit_uir(f"gemm_swiglu_{M}_{K}_{N}", _build_tkn_swiglu(M, K, N), rewrite=True)

def enabled() -> bool: return os.environ.get("TINYKEV_GEMM", "1") != "0"

def matmul(x:Tensor, w:Tensor) -> Tensor|None:
  """`x @ w.T` with w [N,K] (a Linear weight), or None when the fast path does not apply."""
  if not enabled() or x.dtype is not dtypes.float16 or w.dtype is not dtypes.float16 or x.device != w.device: return None
  K, N = w.shape[1], w.shape[0]
  M = math.prod(x.shape[:-1])
  if M < 32 or K == 0 or N == 0 or _plan(M, K, N) is None: return None
  try: kern = get_gemm(M, K, N)
  except Exception: return None
  out = UOp.alloc((M, N), dtypes.float16, device=x.device)
  return Tensor(out.after(kern.call(out, x.reshape(M, K).uop, w.uop))).reshape(*x.shape[:-1], N)

def matmul_t(x:Tensor, wt:Tensor) -> Tensor|None:
  """`x @ wt` with pre-transposed wt [K, N], or None when the fast path does not apply."""
  if not enabled() or x.dtype is not dtypes.float16 or wt.dtype is not dtypes.float16 or x.device != wt.device: return None
  K, N = wt.shape[0], wt.shape[1]
  M = math.prod(x.shape[:-1])
  if M < 16 or K == 0 or N == 0: return None
  if (cfg := TKN_SHAPES.get((K, N))) is not None:
    BM, BN, BK, WM, WN, pad = cfg
    THREADS = WM * WN * 32
    if M % BM == 0 and N % BN == 0 and K % BK == 0 and (BM * BK // 2) % THREADS == 0 and (BK * BN // 2) % THREADS == 0:
      try:
        kern = get_gemm_tkn(M, K, N, BM, BN, BK, WM, WN, pad)
        out = UOp.alloc((M, N), dtypes.float16, device=x.device)
        return Tensor(out.after(kern.call(out, x.reshape(M, K).uop, wt.uop))).reshape(*x.shape[:-1], N)
      except Exception: pass
  if M < 32 or _plan_kn(M, K, N) is None: return None
  try: kern = get_gemm_kn(M, K, N)
  except Exception: return None
  out = UOp.alloc((M, N), dtypes.float16, device=x.device)
  return Tensor(out.after(kern.call(out, x.reshape(M, K).uop, wt.uop))).reshape(*x.shape[:-1], N)

def matmul_t_splitk(x:Tensor, wt:Tensor) -> Tensor|None:
  """Split-K `x @ wt` returning `(SK, M, N)` f16 partials for consumption by `add_rmsnorm`, or None."""
  if not enabled() or x.dtype is not dtypes.float16 or wt.dtype is not dtypes.float16 or x.device != wt.device: return None
  K, N = wt.shape[0], wt.shape[1]
  M = math.prod(x.shape[:-1])
  if M < 16 or (cfg := TKN_SPLITK.get((K, N))) is None: return None
  BM, BN, BK, WM, WN, pad, sk = cfg
  THREADS = WM * WN * 32
  if sk <= 1 or M % BM != 0 or N % BN != 0 or K % (BK * sk) != 0 \
      or (BM * BK // 2) % THREADS != 0 or (BK * BN // 2) % THREADS != 0: return None
  try:
    kern = get_gemm_tkn(M, K, N, BM, BN, BK, WM, WN, pad, sk)
    out = UOp.alloc((sk, M, N), dtypes.float16, device=x.device)
    return Tensor(out.after(kern.call(out, x.reshape(M, K).uop, wt.uop)))
  except Exception: return None

def swiglu_matmul_t(x:Tensor, wgt:Tensor, wut:Tensor) -> Tensor|None:
  """Fused `silu(x @ wgt) * (x @ wut)` with pre-transposed [K, N] weights."""
  if not enabled() or x.dtype is not dtypes.float16 or wgt.dtype is not dtypes.float16 or wut.dtype is not dtypes.float16: return None
  if x.device != wgt.device or x.device != wut.device: return None
  K, N = wgt.shape[0], wgt.shape[1]
  M = math.prod(x.shape[:-1])
  if M < 16 or M % 16 != 0 or N % 32 != 0 or K % 16 != 0: return None
  try: kern = get_swiglu_tkn(M, K, N)
  except Exception: return None
  out = UOp.alloc((M, N), dtypes.float16, device=x.device)
  return Tensor(out.after(kern.call(out, x.reshape(M, K).uop, wgt.uop, wut.uop))).reshape(*x.shape[:-1], N)

def _linear_call(self, x:Tensor) -> Tensor:
  if self.bias is None and getattr(self, "groups", 1) == 1:
    if (wt := getattr(self, "_weight_t", None)) is not None:
      if (out := matmul_t(x, wt)) is not None: return out
      if x.dtype is wt.dtype: return x.linear(wt, None)
    if (w := getattr(self, "weight", None)) is not None and (out := matmul(x, w)) is not None: return out
  return self._orig_call(x)

def install() -> None:
  """Route `tinygrad.llm.model.Linear` through the hand-scheduled GEMMs (idempotent)."""
  from tinygrad.llm.model import Linear
  if getattr(Linear, "_tinykev_gemm", False): return
  Linear._orig_call = Linear.__call__
  Linear.__call__ = _linear_call
  Linear._tinykev_gemm = True
