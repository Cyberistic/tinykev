"""Kev inference on tinygrad: one causal row per question, then the pointer head.

The backbone is `tinygrad.llm.model.Transformer` (the Qwen3.5 hybrid stack tinygrad already runs from GGUF).
Kev needs hidden states, not tokens, so instead of `Transformer.forward` (which samples a token) we walk the
same blocks and keep `output_norm(x)`.  With Gated DeltaNet layers the packed block-causal form is not
expressible, so every question is `state + branch` as one causal sequence starting at position 0 -- Kev's
`forward_rows_batch`.  Rows are independent: the result does not depend on how questions are batched.
"""
from __future__ import annotations

import math

from tinygrad import Tensor, dtypes

from .encode import rows_of
from .weights import Loaded

class Kev:
  def __init__(self, loaded:Loaded, device:str|None=None):
    self.loaded = loaded
    self.model = loaded.model
    self.head = loaded.meta["head"]
    self.temperature = float(loaded.meta.get("temperature", 1.0))
    self.head_dim = int(loaded.meta.get("head_dim", 256))
    self.device = device or Tensor.empty(0).device

  def hidden(self, ids:list[int]) -> Tensor:
    """`[len(ids), hidden]` final-norm hidden states of one causal row."""
    t = Tensor([ids], dtype=dtypes.int32, device=self.device)
    x = self.model.token_embd(t).float()
    idx_mask, pre = None, None
    for blk in self.model.blk:
      x, idx_mask, pre = blk(x, (0,), idx_mask, t, pre)
    if pre is not None: raise NotImplementedError("attention-residual streams are not used by Kev checkpoints")
    return self.model.output_norm(x)[0]

  def pointer_logits(self, h:Tensor, decide:int, opts:list[int]) -> Tensor:
    """Kev's `PointerHead.forward`: (k(h_opt) . q(h_decide)) / sqrt(dp) / temperature.

    The two projections and the dot are a templated UIR kernel when it can be built (see `tinykev.uir`),
    otherwise the same math as Tensor ops."""
    qw, qb, kw, kb = self.head["q.weight"], self.head["q.bias"], self.head["k.weight"], self.head["k.bias"]
    from .uir import readout
    q = h[decide] @ qw.T + qb
    k = h[Tensor(opts, dtype=dtypes.int32, device=self.device)] @ kw.T + kb
    if (z := readout(q.reshape(1, -1), k)) is None:
      z = (k * q).sum(-1)
    z = z * (1.0 / math.sqrt(qw.shape[0]))
    return z if self.temperature == 1.0 else z / self.temperature

  def forward(self, enc:dict) -> list[Tensor]:
    """One logits vector per question (Kev's `forward`)."""
    state_ids, _, rows = rows_of(enc)
    out = []
    for r in rows:
      h = self.hidden(state_ids + r["ids"])
      out.append(self.pointer_logits(h, len(state_ids) + r["decide"], [len(state_ids) + o for o in r["opts"]]))
    return out

  def probs(self, enc:dict) -> list[list[float]]:
    """Softmax over each question's options (Kev's `probs`, the serving miss path)."""
    return [z.softmax(-1).tolist() for z in self.forward(enc)]
