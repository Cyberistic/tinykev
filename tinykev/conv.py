"""The DeltaNet short causal conv as a templated UIR kernel.

tinygrad's `short_conv` assembles a `[state | x | pad]` window and sums four sliced products; on Metal that
turns into several poor kernels (~0.68 ms per layer for Kev-0.8B).  This is the same causal conv, one
elementwise kernel over flattened `(batch, token, channel)` with the tap loop unrolled and the channel
vectorized, plus the same state store the reference returns.

`install` is folded into `PortableGatedDeltaNetBlock` via a temporary `short_conv` redirect, like the scan.
"""
from __future__ import annotations

import functools, math, os, pathlib, threading
from dataclasses import replace

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir
from tinygrad.llm.model import short_conv as _ref_short_conv

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
VEC = int(os.environ.get("TINYKEV_CONV_VEC", "8"))
THREADS = int(os.environ.get("TINYKEV_CONV_THREADS", "128"))
PACKED = os.environ.get("TINYKEV_CONV_PACKED", "1") != "0"    # packed-order taps (no row gather)
LOG2E = math.log2(math.e)

def _p(slot:int, shape:tuple[int, ...], dt:dtypes.DType) -> UOp: return UOp.param(slot, dt, shape)

def _build_conv(B:int, T:int, C:int, K:int) -> UOp:
  N = B * T * C
  out, x = _p(0, (B, T, C), dtypes.float32), _p(1, (B, T, C), dtypes.float16)
  state, w = _p(2, (B, K-1, C), dtypes.float32), _p(3, (C, K), dtypes.float16)
  g, tid, r = (UOp.range(N // (VEC * THREADS), 0, AxisType.GLOBAL), UOp.range(THREADS, 1, AxisType.LOCAL),
               UOp.range(VEC, 2, AxisType.UPCAST))
  n = (g * THREADS + tid) * VEC + r
  c, bt = n % C, n // C
  b, t = bt // T, bt % T
  acc = UOp.const(0, dtypes.float32)
  for i in range(K):
    wi = t + i                                        # window index: state rows first, then x
    sv = state[b, wi.minimum(K - 2), c].load()        # already zeroed by the caller when initial
    xv = x[b, (wi - (K - 1)).maximum(0), c].load()
    acc = acc + (wi < (K - 1)).where(sv, xv.float()) * w[c, i].load().float()
  silu = acc * ((-acc * LOG2E).exp2() + 1.0).reciprocal()
  return out[b, t, c].store(silu).end(g, tid, r).sink(
    arg=KernelInfo(name="short_conv_uir", opts_to_apply=()))

def _build_conv_packed(T:int, C:int, K:int) -> UOp:
  """The same conv over the packed token order, with a per-token source table for the taps before it.

  Packed tokens are `state + branch_0 + ...`, so a branch's first three taps are the last three prefix tokens,
  not the packed token before it.  `src[t, j]` is the packed index of the token three-back-plus-j inside t's own
  row, or `T` where the window leaves the row (a fresh row starts from a zeroed conv state)."""
  M = T * C
  out, x = _p(0, (T, C), dtypes.float32), _p(1, (T, C), dtypes.float16)
  w, src = _p(2, (C, K), dtypes.float16), _p(3, (T, K-1), dtypes.int32)
  g, tid, r = (UOp.range(M // (VEC * THREADS), 0, AxisType.GLOBAL), UOp.range(THREADS, 1, AxisType.LOCAL),
               UOp.range(VEC, 2, AxisType.UPCAST))
  n = (g * THREADS + tid) * VEC + r
  c, t = n % C, n // C
  acc = UOp.const(0, dtypes.float32)
  for j in range(K - 1):
    idx = src[t, j].load()
    xv = x[idx.minimum(T - 1), c].load()
    acc = acc + (idx < T).where(xv.float(), 0.0) * w[c, j].load().float()
  acc = acc + x[t, c].load().float() * w[c, K-1].load().float()
  silu = acc * ((-acc * LOG2E).exp2() + 1.0).reciprocal()
  return out[t, c].store(silu).end(g, tid, r).sink(arg=KernelInfo(name="short_conv_packed", opts_to_apply=()))

@functools.cache
def get_conv_packed(T:int, C:int, K:int) -> UOp:
  with Context(SPEC=0):
    u = graph_rewrite(_build_conv_packed(T, C, K), symbolic_simple, enter_calls=True)
    uir = render_uir(u, color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"conv_packed_{T}_{C}_{K}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def short_conv_packed(x:Tensor, weight:Tensor, src:Tensor) -> Tensor|None:
  """(N, C) packed projection -> (N, C) f32 conv+SiLU, or None when the shape does not tile."""
  T, C = x.shape
  K = weight.shape[1]
  if not (enabled() and PACKED) or C % VEC or (T * C) % (VEC * THREADS) or x.dtype is not dtypes.float16: return None
  kern = get_conv_packed(T, C, K)
  out = UOp.alloc((T, C), dtypes.float32, device=x.device)
  return Tensor(out.after(kern.call(out, x.uop, weight.uop, src.uop)))

@functools.cache
def get_conv(B:int, T:int, C:int, K:int) -> UOp:
  key = f"conv_{B}_{T}_{C}_{K}"
  with Context(SPEC=0):
    u = graph_rewrite(_build_conv(B, T, C, K), symbolic_simple, enter_calls=True)
    uir = render_uir(u, color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"{key}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def enabled() -> bool: return os.environ.get("TINYKEV_CONV_UIR", "1") != "0"

def short_conv(state:Tensor, x:Tensor, weight:Tensor, initial:Tensor, dilation:int=1) -> tuple[Tensor, UOp]:
  """Drop-in for `tinygrad.llm.model.short_conv` on a Metal f16 row."""
  B, T, C = x.shape
  K = weight.shape[1]
  if not enabled() or dilation != 1 or T < K-1 or C % VEC != 0 or x.dtype is not dtypes.float16 \
      or state.dtype is not dtypes.float32 or (T*B*C) % (VEC * THREADS) != 0:
    return _ref_short_conv(state, x, weight, initial, dilation)
  kern = get_conv(B, T, C, K)
  out = UOp.alloc((B, T, C), dtypes.float32, device=x.device)
  call = kern.call(out, x.uop, initial.where(0, state).contiguous().uop, weight.uop)
  # the next state is the last K-1 real rows (T >= K-1), the same slice rolling_window stores
  return Tensor(out.after(call)), state.uop.store(x[:, T-K+1:T].cast(state.dtype).uop)
