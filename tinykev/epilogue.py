"""Fused DeltaNet epilogue: gated RMSNorm over the value dim, SiLU gate, cast, and the [B,T,H*V] relayout.

The block does this as `(ssm_norm(core.transpose(1,2)) * out_gate.silu()).cast(dtype).contiguous()`, which on
Metal is several elementwise kernels over a 1.6 MB f32 tensor.  One program per (b, t, head); 32 lanes split V
and the RMS is one shuffle reduce.
"""
from __future__ import annotations

import functools, math, os, pathlib, threading

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

from .gated_delta import _warp_sum

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
LOG2E = math.log2(math.e)

def _p(slot:int, shape:tuple[int, ...], dt:dtypes.DType) -> UOp: return UOp.param(slot, dt, shape)
def _silu(x:UOp) -> UOp: return x * ((-x * LOG2E).exp2() + 1.0).reciprocal()

def _build(B:int, T:int, H:int, V:int, eps:float) -> UOp:
  z = _p(0, (B, T, H*V), dtypes.float16)
  core, gate, w = _p(1, (B, H, T, V), dtypes.float32), _p(2, (B, T, H, V), dtypes.float16), _p(3, (V,), dtypes.float16)
  p = UOp.range(B*T*H, 0, AxisType.GLOBAL)
  lane, ci = UOp.range(32, 1, AxisType.WARP), UOp.range(V // 32, 2, AxisType.UPCAST)
  b, rem = p // (T*H), p % (T*H)
  t, h = rem // H, rem % H
  col = lane + ci * 32
  c = core[b, h, t, col].load()
  scale = (_warp_sum((c*c).reduce(ci, arg=(Ops.ADD, 0))) * (1.0 / V) + eps).rsqrt()
  g = _silu(gate[b, t, h, col].load().float())
  return z[b, t, h*V + col].store((c * scale * w[col].load().float() * g).cast(dtypes.float16)).end(p, lane, ci).sink(
    arg=KernelInfo(name="gated_norm_epilogue", opts_to_apply=()))

@functools.cache
def get_epilogue(B:int, T:int, H:int, V:int, eps:float) -> UOp:
  with Context(SPEC=0):
    u = graph_rewrite(_build(B, T, H, V, eps), symbolic_simple, enter_calls=True)
    uir = render_uir(u, color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"epilogue_{B}_{T}_{H}_{V}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def enabled() -> bool: return os.environ.get("TINYKEV_EPILOGUE", "0") != "0"

def epilogue(core:Tensor, gate:Tensor, weight:Tensor, eps:float) -> Tensor|None:
  B, H, T, V = core.shape
  if not enabled() or core.dtype is not dtypes.float32 or gate.dtype is not dtypes.float16 or V % 32 or (B*T*H) == 0: return None
  kern = get_epilogue(B, T, H, V, eps)
  out = UOp.alloc((B, T, H*V), dtypes.float16, device=core.device)
  return Tensor(out.after(kern.call(out, core.uop, gate.uop, weight.uop)))
