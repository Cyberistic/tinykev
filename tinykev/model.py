"""Kev inference on tinygrad: hidden states of the Qwen3.5 backbone, then the pointer head.

The backbone is `tinygrad.llm.model.Transformer`.  Kev needs hidden states, not tokens, so instead of
`Transformer.forward` (which samples a token) we walk the same blocks and keep `output_norm(x)`.

`probs` uses the packed form by default (`tinykev.packed`: every token once, rows only inside the sequence
mixers).  `TINYKEV_PACKED=0` selects the row form: every question is `state + branch` as one causal row, all rows
right-padded into one batch -- Kev's `forward_rows_batch`.
"""
from __future__ import annotations

import array, math, os

from tinygrad import Device, Tensor, TinyJit, dtypes

from .encode import rows_of
from .weights import Loaded, fuse_attn_qkv, fuse_delta_proj, prepare_linears

class Kev:
  def __init__(self, loaded:Loaded, device:str|None=None, use_jit:bool=False):
    self.loaded, self.model, self.use_jit = loaded, loaded.model, use_jit
    self.device = device or Tensor.empty(0).device
    self._jits:dict[tuple, TinyJit] = {}
    self._in_bufs:dict[tuple, list[tuple[Tensor, object]]] = {}
    self._cache_sets:dict[int, list] = {}   # per-batch recurrent/KV cache buffers (row form)
    from .gemm import install as install_gemm
    from .gated_delta import enabled as delta_uir_enabled, install as install_delta_uir
    from .packed import enabled as packed_enabled, install as install_packed
    install_gemm()
    fuse_delta_proj(self.model)
    fuse_attn_qkv(self.model)
    prepare_linears(self.model)
    if delta_uir_enabled(): install_delta_uir(self.model)
    install_packed(self.model)
    self.use_packed = packed_enabled()
    self.head = {k: v.to(self.device) for k, v in loaded.meta["head"].items()}
    self.temperature = float(loaded.meta.get("temperature", 1.0))

  # ---- row form ----

  def hidden_batch(self, rows:list[list[int]], pad_id:int=0) -> Tensor:
    """`[B, L, hidden]` for several causal rows, right-padded to the longest (pads never affect real tokens)."""
    L = max(len(r) for r in rows)
    t = Tensor([r + [pad_id]*(L-len(r)) for r in rows], dtype=dtypes.int32, device=self.device)
    return self._run(t, (len(rows), L))

  def hidden(self, ids:list[int]) -> Tensor:
    """`[len(ids), hidden]` final-norm hidden states of one causal row."""
    return self._run(Tensor([ids], dtype=dtypes.int32, device=self.device), (1, len(ids)))[0]

  def _run(self, t:Tensor, key:tuple) -> Tensor:
    batch = t.shape[0]
    if (cs := self._cache_sets.get(batch)) is not None:
      for blk, name, c in cs: setattr(blk, name, c)
    else:   # lazily allocated caches are sized by the first batch that touches them
      for blk in self.model.blk:
        for name in ("cache_kv", "recurrent_state", "conv_state"):
          if hasattr(blk, name): delattr(blk, name)
    if self.use_jit:
      if (jit := self._jits.get(key)) is None: jit = self._jits.setdefault(key, TinyJit(self._hidden, prune=True))
      ret = jit(t)
    else: ret = self._hidden(t)
    if batch not in self._cache_sets:
      self._cache_sets[batch] = [(blk, name, getattr(blk, name)) for blk in self.model.blk
                                 for name in ("cache_kv", "recurrent_state", "conv_state") if hasattr(blk, name)]
    return ret

  def _hidden(self, t:Tensor) -> Tensor:
    # keep the backbone's dtype: upcasting a fp16 checkpoint makes every GEMM fp32 on Metal (no tensor cores)
    x = self.model.token_embd(t)
    idx_mask, pre = None, None
    for blk in self.model.blk:
      x, idx_mask, pre = blk(x, (0,), idx_mask, t, pre)
    if pre is not None: raise NotImplementedError("attention-residual streams are not used by Kev checkpoints")
    return self.model.output_norm(x)

  def pointer_logits(self, h:Tensor, decide:int, opts:list[int]) -> Tensor:
    """Kev's `PointerHead.forward`: (k(h_opt) . q(h_decide)) / sqrt(dp) / temperature."""
    qw, qb, kw, kb = self.head["q.weight"], self.head["q.bias"], self.head["k.weight"], self.head["k.bias"]
    from .uir import readout
    q = h[decide] @ qw.T + qb
    k = h[Tensor(opts, dtype=dtypes.int32, device=self.device)] @ kw.T + kb
    if (z := readout(q.reshape(1, -1), k)) is None: z = (k * q).sum(-1)
    z = z * (1.0 / math.sqrt(qw.shape[0]))
    return z if self.temperature == 1.0 else z / self.temperature

  def forward(self, enc:dict) -> list[Tensor]:
    """One logits vector per question (Kev's `forward`), row form: every question is a full `state + branch` row."""
    state_ids, _, rows = rows_of(enc)
    off = len(state_ids)
    hs = self.hidden_batch([state_ids + r["ids"] for r in rows])
    return [self.pointer_logits(hs[i, :off + len(r["ids"])], off + r["decide"], [off + o for o in r["opts"]])
            for i, r in enumerate(rows)]

  # ---- packed form ----

  def _packed_logits(self, ids:Tensor, rows_idx:Tensor, pack_idx:Tensor, flags:Tensor, conv_src:Tensor,
                     chunk_tok:Tensor, chunk_flags:Tensor, tok_row:Tensor, tok_pos:Tensor, q_tiles:Tensor,
                     ro_idx:Tensor, ro_row:Tensor, ro_pos:Tensor,
                     decide:Tensor, opts:Tensor, owner:Tensor, *prefix_flat:Tensor,
                     R:int, L:int, N:int, n_decide:int, n_opts:int) -> Tensor:
    from .gather import gather_rows
    from .packed import Pack, forward_packed, set_pack
    from .uir import readout
    pk = Pack(R, L, N, rows_idx, pack_idx, flags, conv_src, chunk_tok, chunk_flags, tok_row, tok_pos, q_tiles,
              ro_idx, ro_row, ro_pos)
    prefix_layers = tuple((prefix_flat[i], prefix_flat[i + 1]) for i in range(0, len(prefix_flat), 2)) if prefix_flat else None
    cfg = self.model.config
    pruned = False
    if prefix_layers is not None or (cfg.hc is None and not cfg.attn_res_block_size and not any(hasattr(b, "ple") for b in self.model.blk)):
      h, pruned = forward_packed(self.model, ids, pk, readout_only=True, prefix_layers=prefix_layers)
    else:
      set_pack(self.model, pk)
      try: h = self._hidden(ids)[0]
      finally: set_pack(self.model, None)
    qw, qb, kw, kb = self.head["q.weight"], self.head["q.bias"], self.head["k.weight"], self.head["k.bias"]
    if pruned:
      q = h[:n_decide].float() @ qw.T + qb
      k = h[n_decide:n_decide + n_opts].float() @ kw.T + kb
    else:
      q = gather_rows(h, decide).float() @ qw.T + qb
      k = gather_rows(h, opts).float() @ kw.T + kb
    if (z := readout(q.contiguous(), k.contiguous(), owner)) is None: z = (k * q[owner]).sum(-1)
    z = z * (1.0 / math.sqrt(qw.shape[0]))
    return (z if self.temperature == 1.0 else z / self.temperature).contiguous()

  def _pack_inputs(self, key:tuple, lay:dict) -> list[Tensor]:
    """Reuse pre-allocated unified-memory int32 input buffers per shape key to avoid per-request MTLBuffer churn."""
    R, L, N, NC, M_ro = lay["R"], lay["L"], lay["N"], lay["NC"], lay["M_ro"]
    specs = (
      ((1, N), lay["ids"]),
      ((R * L,), lay["rows_idx"]),
      ((N,), lay["pack_idx"]),
      ((N,), lay["flags"]),
      ((N, 3), [x for row in lay["conv_src"] for x in row]),
      ((NC, len(lay["chunk_tok"][0])), [x for row in lay["chunk_tok"] for x in row]),
      ((NC,), lay["chunk_flags"]),
      ((N,), lay["tok_row"]),
      ((N,), lay["tok_pos"]),
      ((lay["NT"], 3), [x for row in lay["q_tiles"] for x in row]),
      ((M_ro,), lay["ro_idx"]),
      ((M_ro,), lay["ro_row"]),
      ((M_ro,), lay["ro_pos"]),
      ((len(lay["decide"]),), lay["decide"]),
      ((len(lay["opts"]),), lay["opts"]),
      ((len(lay["owner"]),), lay["owner"]),
    )
    if (cached := self._in_bufs.get(key)) is None:
      bufs = [Tensor.zeros(*sh, dtype=dtypes.int32, device=self.device).contiguous() for sh, _ in specs]
      Tensor.realize(*bufs)
      Device[self.device].synchronize()
      views = [t.uop.buffer.ensure_allocated().host for t in bufs]
      if all(v is not None for v in views):
        cached = [(t, v.view(fmt="i")) for t, v in zip(bufs, views)]
        self._in_bufs[key] = cached
      else:
        return [Tensor(vals, dtype=dtypes.int32, device=self.device).reshape(*sh) for sh, vals in specs]
    for (t, mv), (_, vals) in zip(cached, specs):
      mv[:] = array.array("i", vals)
    return [t for t, _ in cached]

  def logits_packed(self, enc:dict) -> list[list[float]]:
    """One logits list per question, every token computed once; one JIT per (rows, row length, packed length) shape."""
    from .packed import layout
    lay = layout(enc)
    R, L, N = lay["R"], lay["L"], lay["N"]
    n_decide, n_opts = len(lay["decide"]), len(lay["opts"])
    key = ("packed", R, L, N, lay["NC"], lay["NT"], lay["M_ro"], n_opts)
    args = self._pack_inputs(key, lay)
    fn = lambda *a: self._packed_logits(*a, R=R, L=L, N=N, n_decide=n_decide, n_opts=n_opts)
    if self.use_jit:
      if (jit := self._jits.get(key)) is None: jit = self._jits.setdefault(key, TinyJit(fn, prune=True))
      z = jit(*args)
    else: z = fn(*args)
    flat, out, i = z.tolist(), [], 0
    for oi in enc["opt_idx"]:
      out.append(flat[i:i+len(oi)]); i += len(oi)
    return out

  def cache_prefix(self, state_ids:list[int]):
    """Run the backbone once on a shared state prefix and return a reusable `PrefixState` (DeltaNet + KV cache)."""
    from .packed import Pack, PrefixState, forward_prefix, layout_prefix
    lay = layout_prefix(state_ids)
    S, R, L, N = lay["S"], lay["R"], lay["L"], lay["N"]
    t = lambda v, sh: Tensor(v, dtype=dtypes.int32, device=self.device).reshape(*sh)
    pk = Pack(R, L, N, t(lay["rows_idx"], (R * L,)), t(lay["pack_idx"], (N,)), t(lay["flags"], (N,)),
              t([x for row in lay["conv_src"] for x in row], (N, 3)),
              t([x for row in lay["chunk_tok"] for x in row], (lay["NC"], len(lay["chunk_tok"][0]))),
              t(lay["chunk_flags"], (lay["NC"],)), t(lay["tok_row"], (N,)), t(lay["tok_pos"], (N,)),
              t([x for row in lay["q_tiles"] for x in row], (lay["NT"], 3)))
    layers = forward_prefix(self.model, t(lay["ids"], (1, N)), pk, S)
    Tensor.realize(*[x for pair in layers for x in pair])
    return PrefixState(S, tuple(state_ids), layers)

  def probs_with_prefix(self, prefix, enc:dict) -> list[list[float]]:
    """Branch-only packed inference reusing a cached `PrefixState` (replayed via `TinyJit`)."""
    from .packed import layout_branches
    if tuple(enc["ids"][:prefix.S]) != prefix.state_ids:
      raise ValueError("encoding state prefix does not match cached PrefixState")
    lay = layout_branches(enc, prefix.S)
    R, L, N = lay["R"], lay["L"], lay["N"]
    n_decide, n_opts = len(lay["decide"]), len(lay["opts"])
    key = ("prefix", prefix.S, R, L, N, lay["NC"], lay["NT"], lay["M_ro"], n_opts)
    args = self._pack_inputs(key, lay) + list(prefix.tensors)
    fn = lambda *a: self._packed_logits(*a, R=R, L=L, N=N, n_decide=n_decide, n_opts=n_opts)
    if self.use_jit:
      if (jit := self._jits.get(key)) is None: jit = self._jits.setdefault(key, TinyJit(fn, prune=True))
      z = jit(*args)
    else: z = fn(*args)
    flat, out, i = z.tolist(), [], 0
    for oi in enc["opt_idx"]:
      zs = flat[i:i+len(oi)]; i += len(oi)
      m = max(zs); e = [math.exp(v - m) for v in zs]; s = sum(e)
      out.append([v / s for v in e])
    return out

  def probs(self, enc:dict) -> list[list[float]]:
    """Softmax over each question's options (Kev's `probs`)."""
    if not self.use_packed: return [z.softmax(-1).tolist() for z in self.forward(enc)]
    out = []
    for zs in self.logits_packed(enc):
      m = max(zs); e = [math.exp(v - m) for v in zs]; s = sum(e)
      out.append([v / s for v in e])
    return out
