"""The fused gated-RMSNorm epilogue against the Tensor-op reference."""
from __future__ import annotations

import os, unittest

from tinygrad import Tensor, dtypes

from tinykev.epilogue import epilogue

class TestEpilogue(unittest.TestCase):
  def test_parity(self):
    for B, T, H, V in ((3, 8, 16, 128), (1, 5, 4, 64)):
      core = Tensor.randn(B, H, T, V).realize()
      gate = Tensor.randn(B, T, H, V, dtype=dtypes.float16).realize()
      w = Tensor.randn(V, dtype=dtypes.float16).realize()
      os.environ["TINYKEV_EPILOGUE"] = "1"
      got = epilogue(core, gate, w, 1e-6)
      os.environ.pop("TINYKEV_EPILOGUE", None)
      self.assertIsNotNone(got)
      ref = (((core.transpose(1, 2).rmsnorm(eps=1e-6)) * w.cast(dtypes.float32) * gate.silu())
             .cast(dtypes.float16).reshape(B, T, H*V).realize())
      scale = max(abs(x) for x in ref.flatten().tolist())
      diff = max(abs(a-b) for a, b in zip(got.realize().flatten().tolist(), ref.flatten().tolist()))
      self.assertLess(diff / scale, 0.01, f"B={B} T={T} H={H} V={V}")

if __name__ == "__main__": unittest.main()
