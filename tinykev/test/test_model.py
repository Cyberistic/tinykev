"""Hidden-state extraction, causality and the pointer head, on a tiny randomly initialised Qwen3.5-shaped model."""
from __future__ import annotations

import os, unittest
from pathlib import Path

from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.model import SSMConfig, Transformer, TransformerConfig

from tinykev.model import Kev
from tinykev.weights import Loaded

def tiny_model() -> Transformer:
  cfg = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=2, n_kv_heads=1, norm_eps=1e-6,
                          vocab_size=64, head_dim=16, v_head_dim=16, rope_theta=10_000.0, rope_dim=8, max_context=64,
                          ssm=SSMConfig(conv_kernel=2, state_size=8, group_count=2, time_step_rank=2, inner_size=16, silu_gate=True),
                          ssm_layers=(True, False), attn_output_gate=True)
  model = Transformer(cfg)
  for name, p in nn.state.get_state_dict(model).items():
    norm = name.endswith(("attn_norm.weight", "ffn_norm.weight", "output_norm.weight", "q_norm.weight", "k_norm.weight", "ssm_norm.weight"))
    p.assign(Tensor.ones(*p.shape, dtype=p.dtype, device=p.device) if norm
             else (Tensor.randn(*p.shape, dtype=p.dtype, device=p.device) * 0.1).realize())
  return model

class TestHidden(unittest.TestCase):
  def setUp(self):
    dev = Tensor.empty(0).device
    head = {"q.weight": Tensor.randn(8, 32, device=dev)*0.1, "q.bias": Tensor.zeros(8, device=dev),
            "k.weight": Tensor.randn(8, 32, device=dev)*0.1, "k.bias": Tensor.zeros(8, device=dev)}
    self.kev = Kev(Loaded(tiny_model(), {"head": head, "temperature": 1.7, "head_dim": 8}, {}, Path(".")), device=dev)

  def test_hidden_shape_and_causality(self):
    ids = [3, 11, 7, 19, 2, 5]
    h = self.kev.hidden(ids)
    self.assertEqual(h.shape, (len(ids), 32))
    h_pre = self.kev.hidden(ids[:4])
    diff = max(abs(a-b) for a, b in zip(h[:4].flatten().tolist(), h_pre.flatten().tolist()))
    self.assertLess(diff, 2e-4, "a token's hidden state must not depend on later tokens")

  def test_pointer_head_matches_tensor_math(self):
    h = self.kev.hidden([3, 11, 7, 19, 2, 5])
    opts, decide = [1, 3, 5], 4
    head = self.kev.head
    os.environ["TINYKEV_UIR"] = "0"
    ref = ((h[Tensor(opts)] @ head["k.weight"].T + head["k.bias"]) * (h[decide] @ head["q.weight"].T + head["q.bias"])).sum(-1)
    ref = ref * (1.0 / 8**0.5) / 1.7
    os.environ["TINYKEV_UIR"] = "1"
    z = self.kev.pointer_logits(h, decide, opts)
    diff = max(abs(a-b) for a, b in zip(z.tolist(), ref.tolist()))
    os.environ.pop("TINYKEV_UIR", None)
    self.assertLess(diff, 1e-4)

if __name__ == "__main__": unittest.main()
