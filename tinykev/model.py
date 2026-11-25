"""Kev inference on tinygrad: one causal row per question, then the pointer head.

The backbone is `tinygrad.llm.model.Transformer` (the Qwen3.5 hybrid stack tinygrad already runs from GGUF).
Kev needs hidden states, not tokens, so instead of `Transformer.forward` (which samples a token) we walk the
same blocks and keep `output_norm(x)`.  With Gated DeltaNet layers the packed block-causal form is not
expressible, so every question is `state + branch` as one causal sequence starting at position 0 -- Kev's
`forward_rows_batch`.  Rows are independent: the result does not depend on how questions are batched.
"""
from __future__ import annotations

import math, os

from tinygrad import Tensor, TinyJit, dtypes

from .encode import rows_of
from .weights import Loaded

class Kev:
  def __init__(self, loaded:Loaded, device:str|None=None, use_jit:bool=False):
    self.loaded = loaded
    self.use_jit = use_jit
    self._jits:dict[tuple, TinyJit] = {}
    self._prefix:tuple[tuple[int, ...], int, list]|None = None
    self._branch_caches:dict[int, list] = {}
    self._small_snaps:list[tuple[object, str, Tensor]] = []
    self._cache_sets:dict[int, list] = {}   # per-batch cache buffers (non-prefix mode)
    # 0: full state+branch rows in one padded batch (best first-request latency)
    # 1: state once + padded branch batch (best repeated requests), 2: state once + per-question branches
    self.use_prefix = int(os.environ.get("TINYKEV_PREFIX", "0"))
    self.group_rows = float(os.environ.get("TINYKEV_GROUP", "0"))  # 0 disables row grouping
    self.model = loaded.model
    self.device = device or Tensor.empty(0).device
    from .gemm import install as install_gemm
    install_gemm()
    from .gated_delta import enabled as delta_uir_enabled, install as install_delta_uir
    self.portable_blocks = install_delta_uir(self.model) if delta_uir_enabled() else 0
    self.head = {k: v.to(self.device) for k, v in loaded.meta["head"].items()}
    self.temperature = float(loaded.meta.get("temperature", 1.0))
    self.head_dim = int(loaded.meta.get("head_dim", 256))

  def hidden_batch(self, rows:list[list[int]], pad_id:int=0) -> Tensor:
    """`[B, L, hidden]` for several causal rows, right-padded to the longest.

    Pads sit after every real token and both layer kinds are causal, so a pad never changes a real token's
    hidden state; one padded pass replaces B launches and gives the GEMMs a larger M."""
    L = max(len(r) for r in rows)
    t = Tensor([r + [pad_id]*(L-len(r)) for r in rows], dtype=dtypes.int32, device=self.device)
    return self._run(t, (-len(rows), L, 0), 0)

  def hidden(self, ids:list[int]) -> Tensor:
    """`[len(ids), hidden]` final-norm hidden states of one causal row.

    With `use_jit`, each row length gets a `TinyJit` after its first two calls.  The DeltaNet layers are
    unrolled one Python step per token, so rebuilding their graph per request costs far more than running
    it; replay fixes that (Kev-0.8B, 101-token request: 5.9 s -> 0.97 s).  The first two calls per shape pay
    the capture instead, which is why one-shot CLI use leaves this off."""
    t = Tensor([ids], dtype=dtypes.int32, device=self.device)
    return self._run(t, (1, len(ids), 0), 0)[0]

  def _run(self, t:Tensor, key:tuple, start_pos:int) -> Tensor:
    batch = t.shape[0]
    if not self.use_prefix:
      if (cs := self._cache_sets.get(batch)) is not None: self._set_caches(cs)
      else: self._reset_caches()
    if self.use_jit:
      if (jit := self._jits.get(key)) is None: jit = self._jits.setdefault(key, TinyJit(lambda tt: self._hidden(tt, start_pos), prune=True))
      ret = jit(t)
    else: ret = self._hidden(t, start_pos)
    if not self.use_prefix and batch not in self._cache_sets: self._cache_sets[batch] = self._grab_caches()
    return ret

  def _hidden(self, t:Tensor, start_pos:int=0) -> Tensor:
    # keep the backbone's dtype: upcasting a fp16 checkpoint makes every GEMM fp32 on Metal (no tensor cores)
    x = self.model.token_embd(t)
    idx_mask, pre = None, None
    for blk in self.model.blk:
      x, idx_mask, pre = blk(x, (start_pos,), idx_mask, t, pre)
    if pre is not None: raise NotImplementedError("attention-residual streams are not used by Kev checkpoints")
    return self.model.output_norm(x)

  # ---- state-prefix reuse: the state runs once, the question branches continue from a copy of its caches ----
  # Kev's serving path does the same (`probs_and_prefix`).  Each pass has its own stable cache buffers so both
  # can be TinyJit-captured; the state buffers are copied into the branch buffers before every branch replay.

  def _cache_specs(self) -> list[tuple[object, str, int]]:
    specs = []
    for blk in self.model.blk:
      if hasattr(blk, "cache_kv"): specs.append((blk, "cache_kv", 1))
      if hasattr(blk, "recurrent_state"): specs.append((blk, "recurrent_state", 0))
      if hasattr(blk, "conv_state"): specs.append((blk, "conv_state", 0))
    return specs

  def clear_prefix(self) -> None:
    """Forget the cached state so the next request runs the state pass again (keeps the state buffers)."""
    if self._prefix is not None: self._set_caches(self._prefix[2])
    self._prefix, self._branch_caches = None, {}

  def _reset_caches(self) -> None:
    """Drop lazily allocated caches so a state pass starts from a B=1 set (only before this Kev's first JIT)."""
    for blk in self.model.blk:
      for name in ("cache_kv", "cache_k", "cache_pools", "cache_kg", "recurrent_state", "conv_state"):
        if hasattr(blk, name): delattr(blk, name)

  def _grab_caches(self) -> list[tuple[object, str, int, Tensor]]:
    return [(blk, name, axis, getattr(blk, name)) for blk, name, axis in self._cache_specs()]

  def _set_caches(self, caches:list[tuple[object, str, int, Tensor]]) -> None:
    for blk, name, _axis, t in caches: setattr(blk, name, t)

  @staticmethod
  def _expand(t:Tensor, batch:int, axis:int) -> Tensor:
    shape = list(t.shape); shape[axis] = batch
    return t.reshape(shape[:axis] + [1] + shape[axis+1:]).expand(tuple(shape)).contiguous()

  def prefix_for(self, state_ids:list[int]) -> int:
    """Run the state once (or reuse the last one) and return its length."""
    key = tuple(state_ids)
    if self.use_prefix and self._prefix is not None and self._prefix[0] == key:
      self._set_caches(self._prefix[2]); return self._prefix[1]
    if self._prefix is not None: self._set_caches(self._prefix[2])   # state buffers are the B=1 set
    elif not self._jits: self._reset_caches()                          # stale caches from another Kev/pass shape
    t = Tensor([state_ids], dtype=dtypes.int32, device=self.device)
    self._run(t, (1, len(state_ids), 0), 0).realize()
    caches = self._grab_caches()
    self._small_snaps = [(blk, name, getattr(blk, name).clone()) for blk, name, _ in self._cache_specs()
                         if name in ("recurrent_state", "conv_state")]
    self._prefix, self._branch_caches = (key, len(state_ids), caches), {}
    return len(state_ids)

  def _branch_rows_hidden(self, start:int, rows:list[list[int]]) -> Tensor:
    """Per-question branches on the state cache: no padding, only the recurrent/conv state is restored.

    The attention KV cache keeps positions 0..start from the state pass and each branch overwrites
    start.. for itself, so it needs no copy."""
    out = []
    for r in rows:
      for blk, name, saved in self._small_snaps: getattr(blk, name).assign(saved)
      t = Tensor([r], dtype=dtypes.int32, device=self.device)
      out.append(self._run(t, (1, len(r), start), start)[0])
    return out

  def _branches_hidden(self, start:int, rows:list[list[int]], pad_id:int=0) -> Tensor:
    batch, L = len(rows), max(len(r) for r in rows)
    created = (bc := self._branch_caches.get(batch)) is None
    if created:
      bc = [(blk, name, axis, self._expand(t, batch, axis)) for blk, name, axis, t in self._prefix[2]]
      self._branch_caches[batch] = bc
    # The attention KV prefix at 0..start is written once when the branch buffers are created and every branch
    # pass only overwrites start.., so repeat requests only need the small recurrent/conv state restored.
    for (_, name, _, dst), (_, _, _, src) in zip(bc, self._prefix[2]):
      if created or name in ("recurrent_state", "conv_state"): dst.assign(src.expand(tuple(dst.shape)))
    self._set_caches(bc)
    t = Tensor([r + [pad_id]*(L-len(r)) for r in rows], dtype=dtypes.int32, device=self.device)
    return self._run(t, (-batch, L, start), start)

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
    """One logits vector per question (Kev's `forward`).

    With prefix reuse the state runs once and the branch rows continue from its caches; otherwise every
    question is a full `state + branch` row, all batched in one padded pass."""
    state_ids, _, rows = rows_of(enc)
    if self.use_prefix == 2:
      start, off = self.prefix_for(state_ids), 0
      hs = self._branch_rows_hidden(start, [r["ids"] for r in rows])
    elif self.use_prefix:
      start = self.prefix_for(state_ids)
      hs, off = self._branches_hidden(start, [r["ids"] for r in rows]), 0
    else:
      start, off = 0, len(state_ids)
      full = [state_ids + r["ids"] for r in rows]
      if self.group_rows > 0:
        order = sorted(range(len(full)), key=lambda i: len(full[i]), reverse=True)
        groups:list[list[int]] = []
        for i in order:
          if groups and len(full[i]) >= self.group_rows * len(full[groups[-1][0]]): groups[-1].append(i)
          else: groups.append([i])
        hs = [None]*len(full)
        for g in groups:
          out = self.hidden_batch([full[i] for i in g])
          for j, i in enumerate(g): hs[i] = out[j]
      else: hs = self.hidden_batch(full)
    out = []
    for i, r in enumerate(rows):
      if self.use_prefix == 2: h = hs[i]
      elif self.use_prefix: h = hs[i, :len(r["ids"])]
      elif isinstance(hs, list): h = hs[i][:off + len(r["ids"])]
      else: h = hs[i, :off + len(r["ids"])]
      out.append(self.pointer_logits(h, off + r["decide"], [off + o for o in r["opts"]]))
    return out

  def probs(self, enc:dict) -> list[list[float]]:
    """Softmax over each question's options (Kev's `probs`, the serving miss path)."""
    return [z.softmax(-1).tolist() for z in self.forward(enc)]
