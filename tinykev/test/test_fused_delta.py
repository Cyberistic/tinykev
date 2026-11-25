"""The fused conv+norm+scan kernel against a Tensor reference built from the block's math."""
from __future__ import annotations

import math
import unittest

from tinygrad import Tensor, dtypes

from tinykev.fused_delta import core

def _reference(proj, cw, cstate, beta, alpha, kstate, initial):
  B, T, C = proj.shape; H, K, V = kstate.shape[1], kstate.shape[3], kstate.shape[2]
  state0 = Tensor.zeros(B, 3, C, dtype=dtypes.float32) if initial else cstate
  win = state0.cat(proj.float(), dim=1)
  acc = win[:, 0:T] * cw[:, 0].float()
  for i in range(1, 4): acc = acc + win[:, i:i+T] * cw[:, i].float()
  conv = acc * ((acc * -math.log2(math.e)).exp2() + 1.0).reciprocal()          # silu
  q = conv[:, :, :H*K].reshape(B, T, H, K).rmsnorm(eps=1e-6/K) * (K ** -0.5) * (K ** -0.5)
  k = conv[:, :, H*K:2*H*K].reshape(B, T, H, K).rmsnorm(eps=1e-6/K) * (K ** -0.5)
  v = conv[:, :, 2*H*K:].reshape(B, T, H, V)
  q, k, v = (z.transpose(1, 2) for z in (q, k, v))
  state = Tensor.zeros(B, H, V, K, dtype=dtypes.float32) if initial else kstate
  outs = []
  for t in range(T):
    s1 = state * alpha[:, :, t].reshape(B, H, 1, 1)
    sk = (s1 * k[:, :, t, None, :]).sum(-1)
    delta = (v[:, :, t] - sk) * beta[:, t][..., None]
    state = s1 + delta[:, :, :, None] * k[:, :, t, None, :]
    outs.append((state * q[:, :, t, None, :]).sum(-1))
  return Tensor.stack(*outs, dim=2)

class TestFusedDelta(unittest.TestCase):
  def test_parity(self):
    for initial in (True, False):
      for B, T, H, V, K in ((1, 5, 2, 8, 32), (2, 3, 2, 16, 32)):
        C = 2*H*K + H*V
        proj = Tensor.randn(B, T, C, dtype=dtypes.float16).realize()
        cw = Tensor.randn(C, 4, dtype=dtypes.float16).realize()
        cstate = Tensor.randn(B, 3, C).realize()
        beta = Tensor.rand(B, T, H).realize()
        alpha = (Tensor.rand(B, H, T) * 0.5 + 0.5).realize()
        kstate = Tensor.randn(B, H, V, K).realize()
        got = core(proj, cw, cstate, beta, alpha, kstate, initial).realize()
        ref = _reference(proj, cw, cstate, beta, alpha, kstate, initial).realize()
        scale = max(abs(x) for x in ref.flatten().tolist())
        diff = max(abs(a-b) for a, b in zip(got.flatten().tolist(), ref.flatten().tolist()))
        self.assertLess(diff / scale, 0.02, f"initial={initial} B={B} T={T} H={H} V={V} K={K}")

if __name__ == "__main__": unittest.main()
