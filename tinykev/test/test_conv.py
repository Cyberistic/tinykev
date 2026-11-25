"""The DeltaNet short conv UIR kernel against tinygrad's reference window implementation."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.llm.model import short_conv as reference

from tinykev.conv import short_conv, short_conv_packed
from tinykev.gather import gather_rows
from tinykev.packed import layout
from tinykev.test.test_packed import _enc

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

  def test_packed_matches_rows(self):
    enc = _enc([1, 5, 9, 2], [([7, 3, 11, 4, 6, 8], [2, 4]), ([12, 13, 14], [0, 1]),
                              ([20, 21, 22, 23, 24], [1, 2, 3])])
    lay = layout(enc)
    R, L, N, C, K = lay["R"], lay["L"], lay["N"], 64, 4
    rows_idx = Tensor(lay["rows_idx"], dtype=dtypes.int32).realize()
    pack_idx = Tensor(lay["pack_idx"], dtype=dtypes.int32).realize()
    src = Tensor(lay["conv_src"], dtype=dtypes.int32).realize()
    x = Tensor.randn(N, C, dtype=dtypes.float16).realize()
    weight = Tensor.randn(C, K, dtype=dtypes.float16).realize()
    got = short_conv_packed(x, weight, src)
    self.assertIsNotNone(got)
    ref_rows, _ = reference(Tensor.zeros(R, K-1, C, dtype=dtypes.float32),
                            gather_rows(x, rows_idx).reshape(R, L, C), weight, Tensor(True))
    ref = gather_rows(ref_rows.reshape(R*L, C), pack_idx).realize()
    # Padded packed slots deliberately differ: gather-back aliases them to row zero, while packed conv computes them.
    real = len(enc["ids"])
    diff = max(abs(a-b) for a, b in zip(got[:real].realize().flatten().tolist(), ref[:real].flatten().tolist()))
    self.assertLess(diff, 2e-2)

if __name__ == "__main__": unittest.main()
