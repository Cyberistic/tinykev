"""The portable gated-delta scan: UIR round-trip and parity with the Tensor recurrence."""
from __future__ import annotations

import unittest

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.uir import parse_uir, render_uir

from tinykev.gated_delta import (ROW_TILE, UIR_DIR, WRITE_STATE, get_gated_delta, get_gated_delta_packed,
                                 kernel_params, packed_kernel_params)

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
  def test_hook_vt_matches_value_dim(self):
    # regression: VT is head_v_dim/ROW_TILE, i.e. v_tiles*4/ROW_TILE (not v_tiles/ROW_TILE)
    self.assertEqual(kernel_params(batch=1, heads=16, tokens=64, v_tiles=32, k_blks=4, alpha_k=1)["VT"], 128 // ROW_TILE)
    self.assertEqual(kernel_params(batch=1, heads=16, tokens=64, v_tiles=32, k_blks=4, alpha_k=1)["V"], 128)
  def test_uir_roundtrip(self):
    from tinygrad.uop.ops import Ops
    u = get_gated_delta()
    self.assertIs(u.op, Ops.SINK)
    self.assertTrue(any(x.is_template for x in u.toposort()))
    p = UIR_DIR / "gated_delta_portable.uir"
    self.assertTrue(p.exists())
    self.assertEqual(render_uir(parse_uir(p.read_text()), color=False).strip(), render_uir(u, color=False).strip())

  def test_packed_save_restore(self):
    from tinykev.packed import _scan_ref
    H, T, K, V = 2, 12, 32, 8
    q, k = Tensor.randn(H,T,K).realize(), Tensor.randn(H,T,K).realize()
    v = Tensor.randn(H,T,V).realize()
    beta, alpha = Tensor.rand(H,T).realize(), Tensor.rand(H,T,1).realize()*0.5+0.5
    # Four prefix tokens, then two branches and three padded steps. Each branch must restart from the prefix state.
    flags = Tensor([0,0,0,1, 2,0,0, 2,0, 0,0,0], dtype=dtypes.int32).realize()
    kq = (q*k).sum(-1).realize()
    kern = get_gated_delta_packed().specialize(**packed_kernel_params(heads=H, tokens=T, k_blks=K//32,
                                                                     v_tiles=V//4, alpha_k=1))
    out = UOp.alloc((H,T,V), dtypes.float32, device=q.device)
    got = Tensor(out.after(kern.call(out, q.uop, k.uop, v.uop, beta.uop, alpha.uop, kq.uop, flags.uop))).realize()
    ref = _scan_ref(q, k, v, beta, alpha, flags).realize()
    diff = max(abs(a-b) for a, b in zip(got.flatten().tolist(), ref.flatten().tolist()))
    self.assertLess(diff, 1e-2)

  def test_matches_recurrence(self):
    for T, K, V in ((1, 32, 8), (5, 32, 8), (4, 128, 128)):
      B, H = 1, 2
      kern = get_gated_delta().specialize(batch=B, heads=H, tokens=T, K=K, V=V, KB=K//32, VT=V//ROW_TILE, alpha_k=1)
      q, k = Tensor.randn(B,H,T,K).realize(), Tensor.randn(B,H,T,K).realize()
      v = Tensor.randn(B,H,T,V).realize()
      beta, alpha = Tensor.rand(B,H,T).realize(), Tensor.rand(B,H,T,1).realize()*0.5+0.5
      state, kq = Tensor.randn(B,H,V,K).realize(), (q*k).sum(-1).realize()
      out = UOp.alloc((B,H,T,V), dtypes.float32, device=q.device)
      got = Tensor(out.after(kern.call(out, q.uop, k.uop, v.uop, beta.uop, alpha.uop, state.uop, kq.uop,
                                      UOp.const(0, dtypes.weakint)))).realize()
      ref, ref_state = _reference(q, k, v, beta, alpha, Tensor.zeros(B,H,V,K).realize())
      d_out = max(abs(a-b) for a, b in zip(got.flatten().tolist(), ref.flatten().tolist()))
      self.assertLess(d_out, 1e-2, f"out T={T} K={K} V={V}")
      if WRITE_STATE:
        d_state = max(abs(a-b) for a, b in zip(state.flatten().tolist(), ref_state.flatten().tolist()))
        self.assertLess(d_state, 1e-2, f"state T={T} K={K} V={V}")

if __name__ == "__main__": unittest.main()
