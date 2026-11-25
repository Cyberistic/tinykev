"""The hand-scheduled WMMA GEMM: tile plan and parity with tinygrad's matmul."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes

from tinykev.gemm import _plan, matmul

class TestGemm(unittest.TestCase):
  def test_plan_m_tiles(self):
    self.assertEqual(_plan(192, 1024, 6144), (4, 4, 4))     # the cold row batch
    self.assertEqual(_plan(144, 1024, 6144), (3, 4, 4))     # 18 TC fragments -> mu 3
    self.assertEqual(_plan(126, 1024, 6144)[0], 4)          # TC pads 126 -> 128, 16 fragments -> mu 4
    self.assertIsNone(_plan(8, 1024, 6144))                 # too small for the tiling

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

if __name__ == "__main__": unittest.main()
