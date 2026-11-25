"""Row gather `out[i, :] = src[idx[i], :]` as a UIR kernel.

Packed inference moves activations between the packed token order (state once, then every branch) and the
padded row order the sequence mixers need (`state + branch` per question).  tinygrad lowers `src[idx]` with a
Tensor index to a one-hot compare-and-reduce over every source row; this is one vectorized load per element.
"""
from __future__ import annotations

import functools, os, pathlib, threading

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.uop.uir import parse_uir, render_uir

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
VEC = 8

def _threads(cb:int) -> int: return next(t for t in (64, 32, 16, 8, 4, 2, 1) if cb % t == 0)

def _build(n_out:int, n_in:int, C:int, dt) -> UOp:
  out, src, idx = UOp.param(0, dt, (n_out, C)), UOp.param(1, dt, (n_in, C)), UOp.param(2, dtypes.int32, (n_out,))
  th = _threads(C // VEC)
  i, cb, t, r = (UOp.range(n_out, 0, AxisType.GLOBAL), UOp.range(C // (VEC * th), 1, AxisType.GLOBAL),
                 UOp.range(th, 2, AxisType.LOCAL), UOp.range(VEC, 3, AxisType.UPCAST))
  c = (cb * th + t) * VEC + r
  return out[i, c].store(src[idx[i].load(), c].load()).end(i, cb, t, r).sink(arg=KernelInfo(name="row_gather", opts_to_apply=()))

@functools.cache
def get_gather(n_out:int, n_in:int, C:int, dt) -> UOp:
  with Context(SPEC=0): uir = render_uir(_build(n_out, n_in, C, dt), color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"gather_{n_out}_{n_in}_{C}_{dt.name}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def enabled() -> bool: return os.environ.get("TINYKEV_GATHER_UIR", "1") != "0"

def gather_rows(src:Tensor, idx:Tensor) -> Tensor:
  """`src[idx]` for a 2D `src` and a 1D int32 `idx`."""
  n_in, C = src.shape
  n_out = idx.shape[0]
  if not enabled() or C % VEC or src.dtype not in (dtypes.float16, dtypes.float32): return src[idx]
  kern = get_gather(n_out, n_in, C, src.dtype)
  out = UOp.alloc((n_out, C), src.dtype, device=src.device)
  return Tensor(out.after(kern.call(out, src.contiguous().uop, idx.uop)))
