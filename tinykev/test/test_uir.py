"""The templated UIR readout kernel round-trips through `.uir` and matches the Tensor-op math."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.uop.uir import parse_uir, render_uir

from tinykev.uir import UIR_DIR, fused_pointer_head, get_readout, readout

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

  def test_fused_pointer_head(self):
    M_ro, D, DP, n_dec, n_opt, scale = 16, 1024, 256, 3, 8, 1.0 / 16.0
    h = (Tensor.randn(M_ro, D, dtype=dtypes.half) * 0.2).realize()
    qw = (Tensor.randn(DP, D) * 0.1).realize()
    qb = Tensor.randn(DP).realize()
    kw = (Tensor.randn(DP, D) * 0.1).realize()
    kb = Tensor.randn(DP).realize()
    owner = Tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=dtypes.int32).realize()
    got = fused_pointer_head(h, qw, qb, kw, kb, owner, n_dec, n_opt, scale)
    self.assertIsNotNone(got)
    q = h[:n_dec].float() @ qw.T + qb
    k = h[n_dec:n_dec + n_opt].float() @ kw.T + kb
    ref = ((k * q[owner]).sum(-1) * scale).realize()
    self.assertLess((got.realize() - ref).abs().max().item(), 1e-3)

if __name__ == "__main__": unittest.main()
