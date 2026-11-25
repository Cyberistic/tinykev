"""Portable gated-delta-rule scan, written as a templated UIR kernel in the `slopkernels` style.

tinygrad's `GatedDeltaNetBlock` only uses a fused scan kernel on AMD (`slopkernels.amd_gated_delta`); everywhere
else it unrolls the recurrence one Python step per token per layer, which turns a Kev request into thousands of
kernel launches.  This kernel is the same computation with the same call shape as `amd_gated_delta`, but it does
not use AMD warp intrinsics: one program per (batch, head, value row) owns the K x 1 strip of the recurrent state
in registers, the key dimension is an UPCAST vector and the token loop is a LOOP range, so it lowers anywhere.

    for t: S *= alpha; delta = (v[t] - S.k[t]) * beta[t]; S += delta.k[t]; out[t] = S.q[t]

It plugs into tinygrad's existing call site (the `amd_gated_delta` branch of `GatedDeltaNetBlock._attention`) via
`install`, which only redirects that one kernel name for the duration of the block's attention call.
"""
from __future__ import annotations

import functools, os, pathlib, threading
from dataclasses import replace

from tinygrad import Context, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.llm.model import GatedDeltaNetBlock
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, ParamArg, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
ROW_TILE = int(os.environ.get("TINYKEV_ROW_TILE", "8"))   # value rows per program (must divide head_v_dim)
NATIVE_WARP = os.environ.get("TINYKEV_WARP_NATIVE", "1") != "0"
# Kev's row form always starts at position 0 (or restores the state from a snapshot), so the final recurrent
# state is never read back.  Skipping its 128x128 store saves ~25 MB of writes per layer.
WRITE_STATE = os.environ.get("TINYKEV_SCAN_STATE", "0") != "0"

def _t(name:str, lo:int=1, hi:int=1 << 20) -> UOp: return UOp.template(name, (lo, hi))

def _warp_sum(v:UOp) -> UOp:
  """Sum a value across the 32 lanes with Metal's SIMD-group shuffle (no shared memory, no barrier).

  A reduced value still carries RANGE axes, so materialize it into a one-element register first (the same
  trick slopkernels' warp_reduce uses) and shuffle the shaped value."""
  if not NATIVE_WARP: return v.reduce(UOp.range(32, 1, AxisType.WARP), arg=(Ops.ADD, 0))
  if v.op is Ops.REDUCE or any(u.op is Ops.REDUCE for u in v.src):
    tmp = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
    v = tmp.after(tmp[0].store(v))[0].load()
  for off in (16, 8, 4, 2, 1):
    sh = UOp.native(f"simd_shuffle_xor({{0}}, {off})", shape=v.shape, dtype=dtypes.int32, fmt=True)
    v = v + sh.call(v.bitcast(dtypes.int32)).bitcast(dtypes.float32)
  return v.reshape(())
def _p(slot:int, shape:tuple[int|UOp, ...], dt:dtypes.DType) -> UOp: return UOp.param(slot, dt, shape)
def _v(slot:int, name:str) -> UOp:
  return UOp.variable(name, 0, 1 << 20).replace(arg=ParamArg(slot, dtypes.weakint, (0, 1 << 20), name, AddrSpace.ALU, None))

def _build_gated_delta() -> UOp:
  """slopkernels' `gated_delta_prefill` layout, with the ds_swizzle warp reduce replaced by a WARP-axis reduce.

  One program per (batch, head, 4 value rows); 32 lanes split the key dimension, each lane owns 4 state values
  per row in registers.  The dot products are a REDUCE over the UPCAST key lanes + a REDUCE over the WARP axis,
  which codegen lowers to a local buffer and a cross-thread reduction (barriers included)."""
  batch, heads, tokens, K, V, KB, VT, alpha_k = (_t("batch"), _t("heads"), _t("tokens"), _t("K", 32, 4096),
                                                 _t("V", 4, 4096), _t("KB", 1, 128), _t("VT", 1, 1024), _t("alpha_k"))
  row_tile = ROW_TILE
  out, q, k, v = (_p(0, (batch, heads, tokens, V), dtypes.float32), _p(1, (batch, heads, tokens, K), dtypes.float32),
                  _p(2, (batch, heads, tokens, K), dtypes.float32), _p(3, (batch, heads, tokens, V), dtypes.float32))
  beta, alpha = _p(4, (batch, heads, tokens), dtypes.float32), _p(5, (batch, heads, tokens, alpha_k), dtypes.float32)
  state, kq = _p(6, (batch, heads, V, K), dtypes.float32), _p(7, (batch, heads, tokens), dtypes.float32)
  start_pos = _v(8, "start_pos")

  bh_row = UOp.range(batch * heads * VT, 0, AxisType.GLOBAL)
  bh, row_base = bh_row // VT, (bh_row % VT) * row_tile
  b, h = bh // heads, bh % heads
  ci, lane = UOp.range(KB, 4, AxisType.UPCAST), UOp.range(32, 1, AxisType.WARP)
  col = lane + ci * 32
  reg = UOp.alloc((row_tile, KB), dtypes.float32, addrspace=AddrSpace.REG)
  r_init = UOp.range(row_tile, 3, AxisType.UPCAST)
  init = start_pos.eq(0).where(0.0, state[b, h, row_base + r_init, col].load().float())   # restart -> zero state
  reg = reg.after(reg[r_init, ci].store(init).end(ci, r_init))

  token = UOp.range(tokens, 2)                                                            # serial scan
  prev = reg.after(token)
  kv, qv = k[b, h, token, col].load(), q[b, h, token, col].load()
  av, bv, kqv = alpha[b, h, token, col % alpha_k].load(), beta[b, h, token].load(), kq[b, h, token].load()
  if NATIVE_WARP:
    # shuffle reduce: one Python step per value row, no shared memory / barrier across lanes
    stores = []
    for r_i in range(row_tile):
      row = row_base + r_i
      decayed = prev[r_i, ci].load() * av
      state_k = _warp_sum((decayed * kv).reduce(ci, arg=(Ops.ADD, 0)))
      state_q = _warp_sum((decayed * qv).reduce(ci, arg=(Ops.ADD, 0)))
      delta = (v[b, h, token, row].load() - state_k) * bv
      stores += [out[b, h, token, row.valid(lane.eq(0))].store(state_q + delta * kqv),
                 reg[r_i, ci].store(decayed + delta * kv).end(ci)]
    step = UOp.group(*stores).end(token)
  else:
    r_v = UOp.range(row_tile, 6, AxisType.UPCAST)
    decayed = prev[r_v, ci].load() * av                                                   # (rows, K/32, 32)
    state_k = (decayed * kv).reduce(ci, arg=(Ops.ADD, 0)).reduce(lane, arg=(Ops.ADD, 0))
    state_q = (decayed * qv).reduce(ci, arg=(Ops.ADD, 0)).reduce(lane, arg=(Ops.ADD, 0))
    delta = (v[b, h, token, row_base + r_v].load() - state_k) * bv
    step = UOp.group(out[b, h, token, (row_base + r_v).valid(lane.eq(0))].store(state_q + delta * kqv),
                     reg[r_v, ci].store(decayed + delta * kv).end(ci, r_v)).end(token)

  if not WRITE_STATE: return step.end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_portable", opts_to_apply=()))
  r_out = UOp.range(row_tile, 5, AxisType.UPCAST)
  return state[b, h, row_base + r_out, col].store(reg.after(step)[r_out, ci].load()).end(ci, r_out, lane, bh_row).sink(
    arg=KernelInfo(name="gated_delta_portable", opts_to_apply=()))

@functools.cache
def get_gated_delta() -> UOp:
  """Generate `tinykev/uir/gated_delta_portable.uir` fresh and return the parsed templated UOp."""
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  with Context(SPEC=0):
    u = graph_rewrite(_build_gated_delta(), symbolic_simple, enter_calls=True)
    uir = render_uir(u, color=False)
  dst = UIR_DIR / "gated_delta_portable.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

# ---- hook into tinygrad's existing DeltaNet kernel call ----

class _Stash:
  """Stand-in projection: returns its slice of a projection already computed for the whole block."""
  def __init__(self, t:UOp, a:int, b:int): self.t, self.a, self.b = t, a, b
  def __call__(self, _x): return self.t[..., self.a:self.b]

class PortableGatedDeltaNetBlock(GatedDeltaNetBlock):
  """`GatedDeltaNetBlock` that runs its scan through the portable UIR kernel.

  The block already calls `_run_kernel("amd_gated_delta", ...)` behind an `_amd_target(device) is not None`
  guard.  For the duration of that call we let the guard pass and route only that kernel name to the portable
  UIR kernel; every other `_amd_target` user (flash attention, etc.) stays off on non-AMD devices."""

  def _attention(self, x, start_pos, idx_mask=None):
    import tinygrad.llm.model as M
    if getattr(self, "_proj", None) is not None and not self.config.ssm.kda:
      # one [T,1024] x [1024, 8224] GEMM instead of qkv + gate + beta + alpha (kev.fused_qwen35 does this on CUDA).
      # N=8224 tiles far better on Metal than N=6144: ~3.4 vs ~1.6 TFLOPS in isolation.
      a, b, c = self._proj_ends
      p = self._proj(x.half())
      saved = (self.attn_qkv, self.attn_gate, self.ssm_beta, self.ssm_alpha)
      self.attn_qkv, self.attn_gate, self.ssm_beta, self.ssm_alpha = (_Stash(p, 0, a), _Stash(p, a, b),
                                                                      _Stash(p, b, c), _Stash(p, c, p.shape[-1]))
      try: return super()._attention(x, start_pos, idx_mask)
      finally: self.attn_qkv, self.attn_gate, self.ssm_beta, self.ssm_alpha = saved
    if enabled() and _is_metal(x.device):
      from . import conv as C
      old_target, old_get, old_conv = M._amd_target, M._get_kernel, M.short_conv
      M._amd_target = lambda dev: (12,)
      if C.enabled(): M.short_conv = C.short_conv
      def _runtime_kernel(name, **params):
        # the block speaks (v_tiles, k_blks); the template is (V, K)
        return get_gated_delta().specialize(**kernel_params(**params)) \
          if name == "amd_gated_delta" else None
      M._get_kernel = _runtime_kernel
      try: return super()._attention(x, start_pos, idx_mask)
      finally: M._amd_target, M._get_kernel, M.short_conv = old_target, old_get, old_conv
    return super()._attention(x, start_pos, idx_mask)

def _is_metal(device) -> bool:
  d = device if isinstance(device, str) else (device[0] if isinstance(device, tuple) and device else None)
  return isinstance(d, str) and d.split(":")[0] == "METAL"

def kernel_params(**params:int) -> dict:
  """Block call parameters -> kernel template values. `v_tiles` is head_v_dim/4; the kernel's VT is head_v_dim/ROW_TILE."""
  return dict(batch=params["batch"], heads=params["heads"], tokens=params["tokens"], K=params["k_blks"]*32,
              V=params["v_tiles"]*4, KB=params["k_blks"], VT=params["v_tiles"]*4 // ROW_TILE, alpha_k=params["alpha_k"])

def enabled() -> bool: return os.environ.get("TINYKEV_DELTA_UIR", "1") != "0"

def install(model) -> int:
  """Give every GatedDeltaNet block in `model` the portable scan. Returns the number of blocks changed."""
  n = 0
  for blk in model.blk:
    if isinstance(blk, GatedDeltaNetBlock) and not isinstance(blk, PortableGatedDeltaNetBlock):
      blk.__class__ = PortableGatedDeltaNetBlock
      n += 1
  return n
