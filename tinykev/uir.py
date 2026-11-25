"""Kev's pointer-head readout as a templated UIR kernel, written the way `slopkernels` writes its AMD kernels.

`slopkernels/src/amd.py` builds each kernel as a pure UOp graph, renders it to `slopkernels/uir/<name>.uir`
and parses it back; `tinygrad.llm.model` then binds shapes with `specialize()` and calls it with
`kernel.call(out, *args)`.  This module uses the same mechanism for the one Kev-specific op: the readout.
The projections stay in Tensor ops (they are GEMMs), and the kernel fuses the gather of the option rows,
the elementwise product against the `<decide>` query and the reduction:

    logit[m] = sum_p Q[owner[m], p] * K[m, p]

The `.uir` text is generated next to this file (never tracked in git, same rule as slopkernels) and is
ordinary tinygrad IR, so it is portable across backends rather than AMD-only.  `Kev.pointer_logits` falls
back to the same math in Tensor ops if the kernel cannot be built; `TINYKEV_UIR=0` forces that path.
"""
from __future__ import annotations

import functools, os, pathlib, threading
from dataclasses import replace

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"

def _t(name:str, lo:int=1, hi:int=1 << 20) -> UOp: return UOp.template(name, (lo, hi))
def _p(slot:int, shape:tuple[int|UOp, ...], dt:dtypes.DType) -> UOp: return UOp.param(slot, dt, shape)

def _build_readout() -> UOp:
  Q, M, DP = _t("Q", 1), _t("M", 1), _t("DP", 1)
  out, q, k, owner = _p(0, (M,), dtypes.float32), _p(1, (Q, DP), dtypes.float32), _p(2, (M, DP), dtypes.float32), _p(3, (M,), dtypes.int32)
  m, p = UOp.range(M, 0, AxisType.GLOBAL), UOp.range(DP, 1, AxisType.UPCAST)
  z = (q[owner[m].load(), p].load() * k[m, p].load()).reduce(p, arg=(Ops.ADD, 0))
  return out[m].store(z).end(m).sink(arg=KernelInfo(name="pointer_readout", opts_to_apply=()))

@functools.cache
def get_readout() -> UOp:
  """Generate `tinykev/uir/pointer_readout.uir` fresh and return the parsed templated UOp."""
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  with Context(SPEC=0):
    u = graph_rewrite(_build_readout(), symbolic_simple, enter_calls=True)
    if allocs := [v for v in u.toposort(enter_calls=True) if v.op is Ops.ALLOC]:
      u = u.substitute({b: replace(b.arg, slot=-1-i) for i, b in enumerate(allocs)}, enter_calls=True)
      u = u.substitute({b.replace(arg=replace(b.arg, slot=-1-i)): b for i, b in enumerate(allocs)}, enter_calls=True)
    uir = render_uir(u, color=False)
  dst = UIR_DIR / "pointer_readout.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def enabled() -> bool: return os.environ.get("TINYKEV_UIR", "1") != "0"

def readout(q:Tensor, k:Tensor, owner:Tensor|None=None) -> Tensor|None:
  """`sum_p q[owner[m], p] * k[m, p]` fused over all options in one kernel, or None when it is unavailable."""
  if not enabled() or (owner is None and q.shape[0] != 1): return None
  try: kern = get_readout()
  except Exception: return None
  M, DP = k.shape
  kern = kern.specialize(Q=q.shape[0], M=M, DP=DP)
  if owner is None: owner = Tensor.zeros(M, dtype=dtypes.int32, device=k.device)
  out = UOp.alloc((M,), dtypes.float32, device=k.device)
  call = kern.call(out, q.uop, k.uop, owner.uop)
  return Tensor(out.after(call))
