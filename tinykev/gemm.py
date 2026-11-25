"""A hand-scheduled WMMA GEMM for Metal, in the slopkernels style.

tinygrad's heuristic picks a small tile for the Kev projection shapes (M=126-192); the same tensor-core
instruction with an explicit schedule (TC + 4x M upcast + 4x N upcast + 4x local N) measures ~1.5x faster on
Kev-0.8B's `[M,1024] x [1024,6144]` and is installed for every `tinygrad.llm.model.Linear`.

The kernel is built with WEAK ranges (not LOOP): tensor cores can only be attached to parallel/reducible
axes, which is why a naive custom matmul is refused.
"""
from __future__ import annotations

import functools, os, pathlib, threading

from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.uir import parse_uir, render_uir

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
# measured on Metal at M=192: (m upcast, n upcast, n local) per (K, N)
TILES = {(1024, 6144): (4, 4, 4), (1024, 3584): (4, 2, 4), (3584, 1024): (2, 2, 4), (1024, 2048): (4, 2, 8),
         (2048, 1024): (4, 2, 4), (1024, 512): (3, 2, 8), (1024, 4096): (4, 4, 2), (1024, 16): (4, 2, 1)}

def _plan(M:int, K:int, N:int) -> tuple[int, int, int]|None:
  base, (mu, nu, nl) = -(-M // 8), TILES.get((K, N), (4, 4, 4))
  if base % mu: mu = next((m for m in (4, 3, 2) if base % m == 0), None)
  if mu is None: return None
  if N % (8 * nu * nl):
    nu, nl = next(((a, b) for a, b in ((4, 4), (2, 4), (4, 2), (2, 2), (2, 8), (8, 2)) if N % (8 * a * b) == 0), (0, 0))
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

@functools.cache
def get_gemm(M:int, K:int, N:int) -> UOp:
  """Generate and parse `tinykev/uir/gemm_<M>_<K>_<N>.uir` (tensor-core scheduled)."""
  with Context(SPEC=0):
    u = _build(M, K, N)
    uir = render_uir(u, color=False)
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  dst = UIR_DIR / f"gemm_{M}_{K}_{N}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_text(uir + "\n", encoding="utf-8"); os.replace(tmp, dst)
  return parse_uir(uir)

def enabled() -> bool: return os.environ.get("TINYKEV_GEMM", "1") != "0"

def matmul(x:Tensor, w:Tensor) -> Tensor|None:
  """`x @ w.T` with w [N,K] (a Linear weight), or None when the fast path does not apply."""
  if not enabled() or x.dtype is not dtypes.float16 or w.dtype is not dtypes.float16 or x.device != w.device: return None
  K, N = w.shape[1], w.shape[0]
  M = 1
  for d in x.shape[:-1]: M *= d
  # the TC fragment plus the local N split: N=16 (the DeltaNet beta/alpha) is too narrow, M=22 (a prefix
  # state pass) is not divisible by the M upcast; both fall back to tinygrad's schedule
  plan = _plan(M, K, N)
  if M < 32 or K == 0 or N == 0 or plan is None: return None
  try: kern = get_gemm(M, K, N)
  except Exception: return None
  out = UOp.alloc((M, N), dtypes.float16, device=x.device)
  call = kern.call(out, x.reshape(M, K).uop, w.uop)
  return Tensor(out.after(call)).reshape(*x.shape[:-1], N)

def _linear_call(self, x:Tensor) -> Tensor:
  w = getattr(self, "weight", None)
  if w is not None and self.bias is None and getattr(self, 'groups', 1) == 1 and (out := matmul(x, w)) is not None: return out
  return self._orig_call(x)

def install() -> None:
  """Route `tinygrad.llm.model.Linear` through the hand-scheduled GEMM (idempotent)."""
  from tinygrad.llm.model import Linear
  if getattr(Linear, "_tinykev_gemm", False): return
  Linear._orig_call = Linear.__call__
  Linear.__call__ = _linear_call
  Linear._tinykev_gemm = True
