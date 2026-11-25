"""Single-pass Metal UIR RMSNorm kernels: `rmsnorm`, fused `add_rmsnorm`, and DeltaNet `gated_rmsnorm`.

tinygrad lowers `nn.RMSNorm` to two kernels (a sum-of-squares reduce and an elementwise scale), plus a third
kernel for every residual add (`x = x + delta`).  Using Metal's `simd_sum`, these kernels fuse the residual add,
the RMS reduction and the affine scale into one pass (~115 fewer kernels per Kev-0.8B request).
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

def _build_rmsnorm(N:int, D:int, eps:float, fused_add:bool=False, sk:int=1) -> UOp:
  """One threadgroup of 128 threads (4 simdgroups) per row `t in 0..N-1`; coalesced vectorized loads.

  When `fused_add=True` and `sk > 1`, `delta` has shape `(sk, N, D)` (Split-K GEMM partials) and is
  reduced into `x` on the fly before normalizing."""
  WAVES = 4
  THREADS = WAVES * 32
  E = D // THREADS
  assert D % THREADS == 0 and sk >= 1
  if fused_add:
    out_sum = UOp.param(0, dtypes.half, (N, D))
    out_norm = UOp.param(1, dtypes.half, (N, D))
    x = UOp.param(2, dtypes.half, (N, D))
    delta = UOp.param(3, dtypes.half, (sk, N, D) if sk > 1 else (N, D))
    w = UOp.param(4, dtypes.half, (D,))
  else:
    out_norm = UOp.param(0, dtypes.half, (N, D))
    x = UOp.param(1, dtypes.half, (N, D))
    w = UOp.param(2, dtypes.half, (D,))

  t = UOp.range(N, 0, AxisType.GLOBAL)
  wid = UOp.range(WAVES, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  tid = wid * 32 + lane
  r = UOp.range(E, 2, AxisType.UPCAST)
  col = tid * E + r

  x_reg = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  if fused_add:
    if sk > 1:
      sv_f = x[t, col].load().float()
      for s in range(sk): sv_f = sv_f + delta[s, t, col].load().float()
      sv_h = sv_f.cast(dtypes.half)
    else:
      sv_h = (x[t, col].load() + delta[t, col].load())
    st_sum = out_sum[t, col].store(sv_h).end(r)
    xv = sv_h.float()
  else:
    st_sum = None
    xv = x[t, col].load().float()

  x_reg = x_reg.after(x_reg[r].store(xv).end(r))
  sq_local = (x_reg[r].load() * x_reg[r].load()).reduce(r, arg=(Ops.ADD, 0))
  w_sq = _simd_sum(sq_local)

  smem = UOp.alloc((WAVES,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  smem = smem.after(smem[wid.valid(lane.eq(0))].store(w_sq).barrier())
  tot = smem[0].load() + smem[1].load() + smem[2].load() + smem[3].load()
  inv = (tot * (1.0 / D) + eps).rsqrt()

  r2 = UOp.range(E, 3, AxisType.UPCAST)
  col2 = tid * E + r2
  nv = (x_reg[r2].load() * inv).cast(dtypes.half) * w[col2].load()
  st_norm = out_norm[t, col2].store(nv).end(r2)
  body = UOp.group(st_sum, st_norm) if st_sum is not None else st_norm
  sfx = f"_sk{sk}" if (fused_add and sk > 1) else ""
  return body.end(lane, wid, t).sink(arg=KernelInfo(name=f"{'add_' if fused_add else ''}rmsnorm_{N}_{D}{sfx}", opts_to_apply=()))

def _build_gated_rmsnorm(N:int, H:int, V:int, P:int, gate_off:int, eps:float, silu_gate:bool=True) -> UOp:
  """DeltaNet epilogue: `rmsnorm(core[h, t, :], w) * silu(p[t, gate_off + h*V : ...]) -> z[t, h*V + v]`.
  1 warp (32 lanes) per `(t, h)`, ZERO shared memory, ZERO barriers, reads `p` directly with stride `P`."""
  E = V // 32
  assert V % 32 == 0
  out = UOp.param(0, dtypes.half, (N, H * V))
  core = UOp.param(1, dtypes.float32, (H, N, V))
  p = UOp.param(2, dtypes.half, (N, P))
  w = UOp.param(3, dtypes.half, (V,))

  th = UOp.range(N * H, 0, AxisType.GLOBAL)
  t, h = th // H, th % H
  lane = UOp.range(32, -1, AxisType.WARP)
  r = UOp.range(E, 2, AxisType.UPCAST)
  v = lane * E + r

  c_reg = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  c_reg = c_reg.after(c_reg[r].store(core[h, t, v].load()).end(r))
  sq = _simd_sum((c_reg[r].load() * c_reg[r].load()).reduce(r, arg=(Ops.ADD, 0)))
  inv = (sq * (1.0 / V) + eps).rsqrt()

  r2 = UOp.range(E, 3, AxisType.UPCAST)
  v2 = lane * E + r2
  normed = c_reg[r2].load() * inv * w[v2].load().float()
  gv = p[t, gate_off + h * V + v2].load().float()
  sig = ((-gv * LOG2E).exp2() + 1.0).reciprocal()
  gate = (gv * sig) if silu_gate else sig
  return out[t, h * V + v2].store((normed * gate).cast(dtypes.half)).end(r2).end(lane, th).sink(
    arg=KernelInfo(name=f"gated_rmsnorm_{N}_{H}_{V}", opts_to_apply=()))

def _build_gather_rmsnorm(V:int, N:int, D:int, eps:float) -> UOp:
  """Fused embedding gather + initial RMSNorm: reads `embd[ids[t], :]` once, emits `(x, rmsnorm(x, w, eps))`."""
  WAVES = 4
  THREADS = WAVES * 32
  E = D // THREADS
  assert D % THREADS == 0
  out_x = UOp.param(0, dtypes.half, (N, D))
  out_h = UOp.param(1, dtypes.half, (N, D))
  embd = UOp.param(2, dtypes.half, (V, D))
  ids = UOp.param(3, dtypes.int32, (N,))
  w = UOp.param(4, dtypes.half, (D,))

  t = UOp.range(N, 0, AxisType.GLOBAL)
  wid = UOp.range(WAVES, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  tid = wid * 32 + lane
  r = UOp.range(E, 2, AxisType.UPCAST)
  col = tid * E + r

  row = ids[t].load().maximum(0).minimum(V - 1)
  xh = embd[row, col].load()
  st_x = out_x[t, col].store(xh).end(r)
  x_reg = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  x_reg = x_reg.after(x_reg[r].store(xh.float()).end(r))
  w_sq = _simd_sum((x_reg[r].load() * x_reg[r].load()).reduce(r, arg=(Ops.ADD, 0)))

  smem = UOp.alloc((WAVES,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  smem = smem.after(smem[wid.valid(lane.eq(0))].store(w_sq).barrier())
  tot = smem[0].load() + smem[1].load() + smem[2].load() + smem[3].load()
  inv = (tot * (1.0 / D) + eps).rsqrt()

  r2 = UOp.range(E, 3, AxisType.UPCAST)
  col2 = tid * E + r2
  st_h = out_h[t, col2].store((x_reg[r2].load() * inv).cast(dtypes.half) * w[col2].load()).end(r2)
  return UOp.group(st_x, st_h).end(lane, wid, t).sink(arg=KernelInfo(name=f"gather_rmsnorm_{N}_{D}", opts_to_apply=()))

@functools.cache
def get_rmsnorm(N:int, D:int, eps:float, fused_add:bool, sk:int=1) -> UOp:
  sfx = f"_sk{sk}" if (fused_add and sk > 1) else ""
  return _emit(f"{'add_' if fused_add else ''}rmsnorm_{N}_{D}{sfx}", _build_rmsnorm(N, D, eps, fused_add, sk))

@functools.cache
def get_gather_rmsnorm(V:int, N:int, D:int, eps:float) -> UOp:
  return _emit(f"gather_rmsnorm_{V}_{N}_{D}", _build_gather_rmsnorm(V, N, D, eps))

@functools.cache
def get_gated_rmsnorm(N:int, H:int, V:int, P:int, gate_off:int, eps:float, silu_gate:bool) -> UOp:
  return _emit(f"gated_rmsnorm_{N}_{H}_{V}_{P}_{gate_off}_{int(silu_gate)}",
               _build_gated_rmsnorm(N, H, V, P, gate_off, eps, silu_gate))

def enabled() -> bool: return os.environ.get("TINYKEV_RMSNORM_UIR", "1") != "0"

def gather_rmsnorm(embd:Tensor, ids:Tensor, w:Tensor, eps:float) -> tuple[Tensor, Tensor]|None:
  """Returns `(embd[ids], rmsnorm(embd[ids], w, eps))` in one Metal UIR kernel, or None when unsupported."""
  V, D = embd.shape
  N = ids.numel()
  if not (enabled() and _is_metal(embd.device)) or embd.dtype is not dtypes.float16 \
      or w.dtype is not dtypes.float16 or ids.dtype is not dtypes.int32 or D % 128 != 0:
    return None
  kern = get_gather_rmsnorm(V, N, D, eps)
  ox = UOp.alloc((N, D), dtypes.half, device=embd.device)
  oh = UOp.alloc((N, D), dtypes.half, device=embd.device)
  call = kern.call(ox, oh, embd.contiguous().uop, ids.reshape(N).contiguous().uop, w.uop)
  return Tensor(ox.after(call)), Tensor(oh.after(call))

def rmsnorm(x:Tensor, w:Tensor, eps:float) -> Tensor:
  """`(N, D)` -> `(N, D)` single-pass UIR RMSNorm on Metal f16, with fallback."""
  N, D = x.shape
  if enabled() and _is_metal(x.device) and x.dtype is dtypes.float16 and w.dtype is dtypes.float16 and D % 128 == 0:
    kern = get_rmsnorm(N, D, eps, False)
    out = UOp.alloc((N, D), dtypes.half, device=x.device)
    return Tensor(out.after(kern.call(out, x.contiguous().uop, w.uop)))
  return x.float().rmsnorm(eps=eps).cast(x.dtype) * w

def add_rmsnorm(x:Tensor, delta:Tensor, w:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  """Returns `(x + delta, rmsnorm(x + delta, w, eps))` in one Metal UIR kernel.

  Also accepts Split-K partials `delta` of shape `(SK, N, D)`, reducing them in-flight."""
  N, D = x.shape
  sk = delta.shape[0] if delta.ndim == 3 else 1
  if enabled() and _is_metal(x.device) and x.dtype is dtypes.float16 and delta.dtype is dtypes.float16 \
      and w.dtype is dtypes.float16 and D % 128 == 0:
    kern = get_rmsnorm(N, D, eps, True, sk)
    o_sum = UOp.alloc((N, D), dtypes.half, device=x.device)
    o_norm = UOp.alloc((N, D), dtypes.half, device=x.device)
    call = kern.call(o_sum, o_norm, x.contiguous().uop, delta.contiguous().uop, w.uop)
    return Tensor(o_sum.after(call)), Tensor(o_norm.after(call))
  d = delta.float().sum(0).cast(x.dtype) if sk > 1 else delta
  s = (x + d).contiguous()
  return s, s.float().rmsnorm(eps=eps).cast(s.dtype) * w

def gated_rmsnorm(core:Tensor, p:Tensor, w:Tensor, gate_off:int, eps:float, silu_gate:bool=True) -> Tensor|None:
  """`core: (H, N, V)` f32, `p: (N, P)` f16, `w: (V,)` f16 -> `(N, H*V)` f16."""
  H, N, V = core.shape
  P = p.shape[-1]
  if not (enabled() and _is_metal(core.device)) or core.dtype is not dtypes.float32 \
      or p.dtype is not dtypes.float16 or w.dtype is not dtypes.float16 or V % 32 != 0:
    return None
  kern = get_gated_rmsnorm(N, H, V, P, gate_off, eps, silu_gate)
  out = UOp.alloc((N, H * V), dtypes.half, device=core.device)
  return Tensor(out.after(kern.call(out, core.contiguous().uop, p.reshape(N, P).contiguous().uop, w.uop)))
