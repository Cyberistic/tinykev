"""The portable gated-delta scan: UIR round-trip and parity with the Tensor recurrence."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.uir import parse_uir, render_uir

from tinykev.gated_delta import UIR_DIR, get_gated_delta

def _reference(q, k, v, beta, alpha, state):
  B, H, T, _ = q.shape
  S, out = state.clone(), []
  for t in range(T):
    S = S * alpha[:, :, t, :].reshape(B, H, 1, alpha.shape[-1])
    sk = (S * k[:, :, t, None, :]).sum(-1)
    delta = (v[:, :, t] - sk) * beta[:, :, t][..., None]
    S = S + delta[:, :, :, None] * k[:, :, t, None, :]
    out.append((S * q[:, :, t, None, :]).sum(-1))
  return Tensor.stack(*out, dim=2), S

class TestPortableGatedDelta(unittest.TestCase):
  def test_uir_roundtrip(self):
    from tinygrad.uop.ops import Ops
    u = get_gated_delta()
    self.assertIs(u.op, Ops.SINK)
    self.assertTrue(any(x.is_template for x in u.toposort()))
    p = UIR_DIR / "gated_delta_portable.uir"
    self.assertTrue(p.exists())
    self.assertEqual(render_uir(parse_uir(p.read_text()), color=False).strip(), render_uir(u, color=False).strip())

  def test_matches_recurrence(self):
    for T, K, V in ((1, 32, 8), (5, 32, 8), (4, 128, 128)):
      B, H = 1, 2
      kern = get_gated_delta().specialize(batch=B, heads=H, tokens=T, K=K, V=V, KB=K//32, VT=V//4, alpha_k=1)
      q, k = Tensor.randn(B,H,T,K).realize(), Tensor.randn(B,H,T,K).realize()
      v = Tensor.randn(B,H,T,V).realize()
      beta, alpha = Tensor.rand(B,H,T).realize(), Tensor.rand(B,H,T,1).realize()*0.5+0.5
      state, kq = Tensor.randn(B,H,V,K).realize(), (q*k).sum(-1).realize()
      out = UOp.alloc((B,H,T,V), dtypes.float32, device=q.device)
      got = Tensor(out.after(kern.call(out, q.uop, k.uop, v.uop, beta.uop, alpha.uop, state.uop, kq.uop,
                                      UOp.const(0, dtypes.weakint)))).realize()
      ref, ref_state = _reference(q, k, v, beta, alpha, Tensor.zeros(B,H,V,K).realize())
      d_out = max(abs(a-b) for a, b in zip(got.flatten().tolist(), ref.flatten().tolist()))
      d_state = max(abs(a-b) for a, b in zip(state.flatten().tolist(), ref_state.flatten().tolist()))
      self.assertLess(d_out, 2e-3, f"out T={T} K={K} V={V}")
      self.assertLess(d_state, 2e-3, f"state T={T} K={K} V={V}")

if __name__ == "__main__": unittest.main()
