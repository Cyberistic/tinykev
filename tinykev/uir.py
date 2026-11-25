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
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"

def _is_metal(device) -> bool:
  d = device if isinstance(device, str) else (device[0] if isinstance(device, tuple) and device else None)
  return isinstance(d, str) and d.split(":")[0] == "METAL"

def _simd_sum(v:UOp) -> UOp:
  if v.op is Ops.REDUCE or any(u.op is Ops.REDUCE for u in v.src):
    tmp = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
    v = tmp.after(tmp[0].store(v))[0].load()
  return UOp.native("simd_sum({0})", shape=v.shape, dtype=dtypes.float32, fmt=True).call(v).reshape(())

def _emit_uir(key:str, u:UOp) -> UOp:
  with Context(SPEC=0):
    uir = render_uir(graph_rewrite(u, symbolic_simple, enter_calls=True), color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"{key}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def _t(name:str, lo:int=1, hi:int=1 << 20) -> UOp: return UOp.template(name, (lo, hi))
def _p(slot:int, shape:tuple[int|UOp, ...], dt:dtypes.DType) -> UOp: return UOp.param(slot, dt, shape)

def _build_readout() -> UOp:
  Q, M, DP = _t("Q", 1), _t("M", 1), _t("DP", 1)
  out, q, k, owner = _p(0, (M,), dtypes.float32), _p(1, (Q, DP), dtypes.float32), _p(2, (M, DP), dtypes.float32), _p(3, (M,), dtypes.int32)
  m, p = UOp.range(M, 0, AxisType.GLOBAL), UOp.range(DP, 1, AxisType.UPCAST)
  z = (q[owner[m].load(), p].load() * k[m, p].load()).reduce(p, arg=(Ops.ADD, 0))
  return out[m].store(z).end(m).sink(arg=KernelInfo(name="pointer_readout", opts_to_apply=()))

def _build_pointer_qk(M_ro:int, D:int, DP:int, n_dec:int, n_opt:int) -> UOp:
  """1 warp per output dim `p in 0..DP-1`: loads `qw[p,:]` & `kw[p,:]` once and dots against all decide/option rows."""
  assert D % 32 == 0
  E = D // 32
  q_out, k_out = _p(0, (n_dec, DP), dtypes.float32), _p(1, (n_opt, DP), dtypes.float32)
  h_in = _p(2, (M_ro, D), dtypes.half)
  qw_p, qb_p = _p(3, (DP, D), dtypes.float32), _p(4, (DP,), dtypes.float32)
  kw_p, kb_p = _p(5, (DP, D), dtypes.float32), _p(6, (DP,), dtypes.float32)
  p, lane = UOp.range(DP, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  e = UOp.range(E, 2, AxisType.UPCAST)
  d = lane * E + e
  wq_r = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  wq_r = wq_r.after(wq_r[e].store(qw_p[p, d].load()).end(e))
  q_bias = qb_p[p].load()
  sts = []
  for qi in range(n_dec):
    dot = _simd_sum((h_in[qi, d].load().float() * wq_r[e].load()).reduce(e, arg=(Ops.ADD, 0)))
    sts.append(q_out[qi, p.valid(lane.eq(0))].store(dot + q_bias))
  wk_r = UOp.alloc((E,), dtypes.float32, addrspace=AddrSpace.REG)
  wk_r = wk_r.after(wk_r[e].store(kw_p[p, d].load()).end(e))
  k_bias = kb_p[p].load()
  for ki in range(n_opt):
    dot = _simd_sum((h_in[n_dec + ki, d].load().float() * wk_r[e].load()).reduce(e, arg=(Ops.ADD, 0)))
    sts.append(k_out[ki, p.valid(lane.eq(0))].store(dot + k_bias))
  return UOp.group(*sts).end(lane, p).sink(arg=KernelInfo(name=f"pointer_qk_{n_dec}_{n_opt}_{DP}", opts_to_apply=()))

def _build_readout_warp(Q:int, M:int, DP:int, scale_val:float) -> UOp:
  """1 warp per option `m`: `simd_sum(q[owner[m], p] * k[m, p]) * scale_val`."""
  assert DP % 32 == 0
  E = DP // 32
  out, q, k, own = _p(0, (M,), dtypes.float32), _p(1, (Q, DP), dtypes.float32), _p(2, (M, DP), dtypes.float32), _p(3, (M,), dtypes.int32)
  m, lane = UOp.range(M, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  e = UOp.range(E, 2, AxisType.UPCAST)
  p = lane * E + e
  ow = own[m].load()
  z = _simd_sum((q[ow, p].load() * k[m, p].load()).reduce(e, arg=(Ops.ADD, 0))) * scale_val
  return out[m.valid(lane.eq(0))].store(z).end(lane, m).sink(arg=KernelInfo(name=f"pointer_readout_warp_{Q}_{M}_{DP}", opts_to_apply=()))

@functools.cache
def get_pointer_qk(M_ro:int, D:int, DP:int, n_dec:int, n_opt:int) -> UOp:
  return _emit_uir(f"pointer_qk_{M_ro}_{D}_{DP}_{n_dec}_{n_opt}", _build_pointer_qk(M_ro, D, DP, n_dec, n_opt))

@functools.cache
def get_readout_warp(Q:int, M:int, DP:int, scale_val:float) -> UOp:
  return _emit_uir(f"pointer_readout_warp_{Q}_{M}_{DP}", _build_readout_warp(Q, M, DP, scale_val))

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

def fused_pointer_head(h_ro:Tensor, qw:Tensor, qb:Tensor, kw:Tensor, kb:Tensor,
                       owner:Tensor, n_dec:int, n_opt:int, scale_val:float) -> Tensor|None:
  """Metal UIR pointer head on pruned `h_ro [M_ro, D]` f16: 2 kernels (`pointer_qk` + `pointer_readout_warp`)."""
  if not (enabled() and _is_metal(h_ro.device)) or h_ro.dtype is not dtypes.float16: return None
  if qw.dtype is not dtypes.float32 or kw.dtype is not dtypes.float32: return None
  M_ro, D = h_ro.shape
  DP = qw.shape[0]
  if D % 32 != 0 or DP % 32 != 0 or n_dec + n_opt > M_ro: return None
  k1 = get_pointer_qk(M_ro, D, DP, n_dec, n_opt)
  k2 = get_readout_warp(n_dec, n_opt, DP, scale_val)
  dev = h_ro.device
  q_buf, k_buf = UOp.alloc((n_dec, DP), dtypes.float32, device=dev), UOp.alloc((n_opt, DP), dtypes.float32, device=dev)
  c1 = k1.call(q_buf, k_buf, h_ro.contiguous().uop, qw.contiguous().uop, qb.contiguous().uop,
               kw.contiguous().uop, kb.contiguous().uop)
  z_buf = UOp.alloc((n_opt,), dtypes.float32, device=dev)
  c2 = k2.call(z_buf, q_buf.after(c1), k_buf.after(c1), owner.contiguous().uop)
  return Tensor(z_buf.after(c2))

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
