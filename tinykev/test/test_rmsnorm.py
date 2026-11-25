"""Single-pass UIR RMSNorm, fused add_rmsnorm, and DeltaNet gated_rmsnorm parity tests."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes
from tinykev.rmsnorm import add_rmsnorm, gated_rmsnorm, rmsnorm

class TestRMSNorm(unittest.TestCase):
  def test_rmsnorm_and_add_rmsnorm(self):
    N, D, eps = 112, 1024, 1e-6
    x = Tensor.randn(N, D, dtype=dtypes.half).realize()
    d = Tensor.randn(N, D, dtype=dtypes.half).realize()
    w = (Tensor.randn(D, dtype=dtypes.half) * 0.2 + 1.0).realize()

    got = rmsnorm(x, w, eps).realize()
    ref = (x.float().rmsnorm(eps=eps).cast(dtypes.half) * w).realize()
    self.assertLess((got.float() - ref.float()).abs().max().item(), 5e-3)

    gs, gn = add_rmsnorm(x, d, w, eps)
    gs, gn = gs.realize(), gn.realize()
    rs = (x + d).realize()
    rn = (rs.float().rmsnorm(eps=eps).cast(dtypes.half) * w).realize()
    self.assertLess((gs.float() - rs.float()).abs().max().item(), 1e-3)
    self.assertLess((gn.float() - rn.float()).abs().max().item(), 5e-3)

    for sk in (2, 4):
      parts = (Tensor.randn(sk, N, D, dtype=dtypes.half) * 0.2).realize()
      ps, pn = add_rmsnorm(x, parts, w, eps)
      ps, pn = ps.realize(), pn.realize()
      rs_sk = (x.float() + parts.float().sum(0)).cast(dtypes.half).realize()
      rn_sk = (rs_sk.float().rmsnorm(eps=eps).cast(dtypes.half) * w).realize()
      self.assertLess((ps.float() - rs_sk.float()).abs().max().item(), 2e-3)
      self.assertLess((pn.float() - rn_sk.float()).abs().max().item(), 5e-3)

  def test_gated_rmsnorm(self):
    H, N, V, P, qa, eps = 16, 112, 128, 8224, 6144, 1e-6
    core = Tensor.randn(H, N, V, dtype=dtypes.float32).realize()
    p = Tensor.randn(N, P, dtype=dtypes.half).realize()
    w = (Tensor.randn(V, dtype=dtypes.half) * 0.2 + 1.0).realize()

    got = gated_rmsnorm(core, p, w, qa, eps, silu_gate=True)
    self.assertIsNotNone(got)
    got = got.realize()
    gate = p[:, qa:qa + H * V].reshape(N, H, V)
    ref = ((core.transpose(0, 1).rmsnorm(eps=eps) * w.float()) * gate.float().silu()).cast(dtypes.half).reshape(N, H * V).realize()
    self.assertLess((got.float() - ref.float()).abs().max().item(), 1e-2)

if __name__ == "__main__": unittest.main()
