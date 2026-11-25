"""The hand-scheduled WMMA GEMM: tile plan and parity with tinygrad's matmul."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes

from tinykev.gemm import _plan, matmul, matmul_t, matmul_t_splitk, swiglu_matmul_t

class TestGemm(unittest.TestCase):
  def test_plan_m_tiles(self):
    self.assertEqual(_plan(192, 1024, 6144), (4, 4, 4))     # the cold row batch
    self.assertEqual(_plan(144, 1024, 6144), (3, 4, 4))     # 18 TC fragments -> mu 3
    self.assertEqual(_plan(126, 1024, 6144)[0], 4)          # TC pads 126 -> 128, 16 fragments -> mu 4
    self.assertIsNone(_plan(8, 1024, 6144))                 # too small for the tiling

  def test_small_m_compiles(self):
    # M=32 keeps only one M tile for mu=4: the axis ends up in the warp fragment and the UPCAST split is refused
    self.assertEqual(_plan(32, 1024, 8224), (2, 2, 2))
    x = Tensor.randn(32, 1024, dtype=dtypes.half).realize()
    w = Tensor.randn(8224, 1024, dtype=dtypes.half).realize()
    self.assertIsNotNone(matmul(x, w).realize())

  def test_matches_matmul(self):
    for M, K, N in ((192, 1024, 6144), (144, 1024, 3584), (192, 3584, 1024), (126, 1024, 512)):
      x = Tensor.randn(M, K, dtype=dtypes.half).realize()
      w = Tensor.randn(N, K, dtype=dtypes.half).realize()
      out = matmul(x, w)
      self.assertIsNotNone(out, f"M={M} K={K} N={N} should take the fast path")
      # the kernel accumulates in f32; compare against an f32 reference (the f16 TC reference has its own error)
      ref = (x.float() @ w.float().T).half().realize()
      scale = max(abs(v) for v in ref.flatten().tolist())
      diff = max(abs(a-b) for a, b in zip(out.realize().flatten().tolist(), ref.flatten().tolist()))
      self.assertLess(diff / scale, 0.05, f"M={M} K={K} N={N}")

  def test_tkn_splitk_and_swiglu(self):
    for M, K, N in ((112, 1024, 8224), (112, 1024, 5120), (112, 2048, 1024), (112, 3584, 1024)):
      x = (Tensor.randn(M, K, dtype=dtypes.half) * 0.2).realize()
      wt = (Tensor.randn(K, N, dtype=dtypes.half) * 0.2).realize()
      ref = (x.float() @ wt.float()).half().realize()
      out = matmul_t(x, wt)
      self.assertIsNotNone(out)
      self.assertLess((out.realize().float() - ref.float()).abs().max().item(), 1e-2, f"tkn {M}x{K}x{N}")
      if (parts := matmul_t_splitk(x, wt)) is not None:
        got = parts.realize().float().sum(0)
        self.assertLess((got - ref.float()).abs().max().item(), 2e-2, f"splitk {M}x{K}x{N}")

    x = (Tensor.randn(112, 1024, dtype=dtypes.half) * 0.2).realize()
    wg = (Tensor.randn(1024, 3584, dtype=dtypes.half) * 0.2).realize()
    wu = (Tensor.randn(1024, 3584, dtype=dtypes.half) * 0.2).realize()
    got = swiglu_matmul_t(x, wg, wu)
    self.assertIsNotNone(got)
    ref = ((x.float() @ wg.float()).silu() * (x.float() @ wu.float())).half().realize()
    self.assertLess((got.realize().float() - ref.float()).abs().max().item(), 2e-2)

if __name__ == "__main__": unittest.main()
