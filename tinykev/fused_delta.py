"""Fused Gated DeltaNet front end: short conv + q/k norm + k_q + recurrence in one UIR kernel.

Per program (batch, head, row tile):
  * the 4-tap causal conv is computed directly from the qkv projection output for this head's q/k and this
    tile's v channels (no `proj -> conv_out` round-trip),
  * q/k are L2-normalised from the conv output, k_q = q.k is reduced in-kernel,
  * the recurrent state update runs as in `gated_delta.py` (lanes split K, 8 value rows/program).

`initial` (start_pos == 0) is a build-time variant: it ignores the conv/recurrent state inputs.  The branch
variant (start_pos > 0) reads the restored conv/recurrent state and never writes them, so the state-pass
snapshot stays pristine.
"""
from __future__ import annotations

import functools, math, os, pathlib, threading

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

from .gated_delta import ROW_TILE, _warp_sum

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
LOG2E = math.log2(math.e)

def _p(slot:int, shape:tuple[int, ...], dt:dtypes.DType) -> UOp: return UOp.param(slot, dt, shape)

def _silu(x:UOp) -> UOp: return x * ((-x * LOG2E).exp2() + 1.0).reciprocal()

def _build(B:int, T:int, H:int, V:int, K:int, initial:bool) -> UOp:
  row_tile, KB = ROW_TILE, K // 32
  C = 2 * H * K + H * V                                     # q | k | v channels
  qbase, kbase, vbase = 0, H * K, 2 * H * K
  out, proj, cw = _p(0, (B, H, T, V), dtypes.float32), _p(1, (B, T, C), dtypes.float16), _p(2, (C, 4), dtypes.float16)
  beta, alpha = _p(3, (B, T, H), dtypes.float32), _p(4, (B, H, T), dtypes.float32)
  cs, ks = _p(5, (B, 3, C), dtypes.float32), _p(6, (B, H, V, K), dtypes.float32)

  bh_row = UOp.range(B * H * (V // row_tile), 0, AxisType.GLOBAL)
  bh, row_base = bh_row // (V // row_tile), (bh_row % (V // row_tile)) * row_tile
  b, h = bh // H, bh % H
  ci, lane = UOp.range(KB, 1, AxisType.UPCAST), UOp.range(32, 0, AxisType.WARP)
  col = lane + ci * 32

  reg = UOp.alloc((row_tile, KB), dtypes.float32, addrspace=AddrSpace.REG)
  r_init = UOp.range(row_tile, 2, AxisType.UPCAST)
  init = UOp.const(0, dtypes.float32) if initial else ks[b, h, row_base + r_init, col].load()
  reg = reg.after(reg[r_init, ci].store(init).end(ci, r_init))

  token = UOp.range(T, 3)
  def wval(wi:UOp, cch:UOp) -> UOp:
    """window value: the previous 3 rows (state) then the projection output."""
    pv = proj[b, (wi - 3).maximum(0), cch].load().float()
    if initial: return (wi >= 3).where(pv, 0.0)
    sv = cs[b, wi.minimum(2), cch].load()
    return (wi < 3).where(sv, pv)

  def conv(cch:UOp) -> UOp:
    acc = UOp.const(0, dtypes.float32)
    for i in range(4): acc = acc + wval(token + i, cch) * cw[cch, i].load().float()
    return _silu(acc)

  q_raw, k_raw = conv(qbase + h * K + col), conv(kbase + h * K + col)
  eps = 1e-6 / K
  qs = _warp_sum((q_raw * q_raw).reduce(ci, arg=(Ops.ADD, 0)))
  ks_ = _warp_sum((k_raw * k_raw).reduce(ci, arg=(Ops.ADD, 0)))
  q = q_raw * ((qs * (1.0 / K) + eps).rsqrt() * (1.0 / K))
  k = k_raw * ((ks_ * (1.0 / K) + eps).rsqrt() * (1.0 / math.sqrt(K)))
  kq = _warp_sum((q * k).reduce(ci, arg=(Ops.ADD, 0)))

  av, bv = alpha[b, h, token].load(), beta[b, token, h].load()
  prev = reg.after(token)
  stores = []
  for r_i in range(row_tile):
    row = row_base + r_i
    v = conv(vbase + h * V + row)
    decayed = prev[r_i, ci].load() * av
    sk = _warp_sum((decayed * k).reduce(ci, arg=(Ops.ADD, 0)))
    sq = _warp_sum((decayed * q).reduce(ci, arg=(Ops.ADD, 0)))
    delta = (v - sk) * bv
    stores += [out[b, h, token, row.valid(lane.eq(0))].store(sq + delta * kq),
               reg[r_i, ci].store(decayed + delta * k).end(ci)]
  step = UOp.group(*stores).end(token)
  return step.end(lane, bh_row).sink(arg=KernelInfo(name="fused_delta", opts_to_apply=()))

@functools.cache
def get_fused(B:int, T:int, H:int, V:int, K:int, initial:bool) -> UOp:
  with Context(SPEC=0):
    u = graph_rewrite(_build(B, T, H, V, K, initial), symbolic_simple, enter_calls=True)
    uir = render_uir(u, color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"fused_delta_{B}_{T}_{H}_{V}_{K}_{int(initial)}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def enabled() -> bool: return os.environ.get("TINYKEV_FUSED_DELTA", "0") != "0"

def core(proj:Tensor, cw:Tensor, cstate:Tensor, beta:Tensor, alpha:Tensor, kstate:Tensor, initial:bool) -> Tensor:
  B, T, C = proj.shape
  H, K, V = kstate.shape[1], kstate.shape[3], kstate.shape[2]
  kern = get_fused(B, T, H, V, K, initial)
  out = UOp.alloc((B, H, T, V), dtypes.float32, device=proj.device)
  args = (proj.uop, cw.uop, beta.uop, alpha.uop) if initial else (proj.uop, cw.uop, beta.uop, alpha.uop, cstate.uop, kstate.uop)
  return Tensor(out.after(kern.call(out, *args)))
