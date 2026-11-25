"""The DeltaNet short conv UIR kernel against tinygrad's reference window implementation."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.llm.model import short_conv as reference

from tinykev.conv import short_conv

class TestShortConv(unittest.TestCase):
  def test_matches_reference(self):
    for initial in (True, False):
      for B, T, C, K in ((2, 8, 32, 4), (1, 5, 16, 4)):
        state = Tensor.randn(B, K-1, C, dtype=dtypes.float32).realize()
        x = Tensor.randn(B, T, C, dtype=dtypes.float16).realize()
        weight = Tensor.randn(C, K, dtype=dtypes.float16).realize()
        init = Tensor(initial).realize()
        ref, _ = reference(state.clone(), x, weight, init)
        got, _ = short_conv(state.clone(), x, weight, init)
        diff = max(abs(a-b) for a, b in zip(got.flatten().tolist(), ref.flatten().tolist()))
        self.assertLess(diff, 2e-2, f"initial={initial} B={B} T={T} C={C}")

if __name__ == "__main__": unittest.main()
