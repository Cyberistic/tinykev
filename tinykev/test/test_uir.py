"""The templated UIR readout kernel round-trips through `.uir` and matches the Tensor-op math."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.uop.uir import parse_uir, render_uir

from tinykev.uir import UIR_DIR, get_readout, readout

class TestReadoutUIR(unittest.TestCase):
  def test_file_and_roundtrip(self):
    from tinygrad.uop.ops import Ops
    u = get_readout()
    self.assertIs(u.op, Ops.SINK)
    self.assertTrue(any(x.is_template for x in u.toposort()))
    p = UIR_DIR / "pointer_readout.uir"
    self.assertTrue(p.exists())
    self.assertNotIn("\x1b", p.read_text())
    self.assertEqual(render_uir(parse_uir(p.read_text()), color=False).strip(), render_uir(u, color=False).strip())

  def test_matches_tensor_math(self):
    q = Tensor.randn(1, 64).realize()
    k = Tensor.randn(7, 64).realize()
    z = readout(q, k)
    self.assertIsNotNone(z)
    ref = (k * q[0]).sum(-1)
    diff = max(abs(a-b) for a, b in zip(z.tolist(), ref.tolist()))
    self.assertLess(diff, 1e-4)

if __name__ == "__main__": unittest.main()
