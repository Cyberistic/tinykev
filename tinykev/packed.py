"""Packed inference: every token once, exactly Kev's packed encoding.

The row form recomputes the state for every question and pads every row to the longest one (Kev-0.8B bench
request: rows of 64/37/44 tokens -> 192 GEMM rows for 101 real tokens).  Here the token-wise work -- embedding,
norms, every projection and the MLPs, ~80 % of the GPU time -- runs on the packed sequence (state once, then each
branch), and only full attention materializes per-question rows:

* Gated DeltaNet: qkv/gate/beta/alpha use one fused projection. The short conv reads each packed token's window
  through a source table, then one packed scan saves the shared prefix state and restores it at each branch.
* Full attention: q/k/v are gathered into rows (RoPE positions 0..L-1, plain causal attention) and the output
  is gathered back before the gate and the output projection.

Every token's computation is the one the row form does, so results match the row form (and Kev's packed
block-causal reference) up to float reassociation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.model import TransformerBlock, apply_rope

from .gather import gather_rows

def enabled() -> bool: return os.environ.get("TINYKEV_PACKED", "1") != "0"
ALIGN = int(os.environ.get("TINYKEV_PACK_ALIGN", "16"))     # packed length bucket (GEMM M tiles)
ROW_ALIGN = int(os.environ.get("TINYKEV_ROW_ALIGN", "8"))   # row length bucket

@dataclass
class Pack:
  R: int              # rows (questions)
  L: int              # padded row length
  N: int              # padded packed length
  rows_idx: Tensor    # (R*L,) packed index of every row slot
  pack_idx: Tensor    # (N,) row slot (r*L + t) of every packed token
  flags: Tensor       # (N,) 0 = step, 1 = save the state after this token, 2 = restore it before
  conv_src: Tensor    # (N, 3) packed index of the three taps before each token inside its own row, N = none
  chunk_tok: Tensor   # (NC, 32) packed token index per 32-token WY chunk, N = pad
  chunk_flags: Tensor # (NC,) 0 = continue, 1 = save prefix state after chunk, 2 = restore before chunk
  tok_row: Tensor     # (N,) owning row index (0..R-1) of each packed token
  tok_pos: Tensor     # (N,) causal position (0..L-1) of each packed token inside its row
  q_tiles: Tensor|None = None  # (NT, 3) [row_r, q_start, q_len] non-overlapping tiles for WMMA Flash Attention
  ro_idx: Tensor|None = None   # (M_ro,) padded readout packed indices (`decide + opts`) for last-layer pruning
  ro_row: Tensor|None = None   # (M_ro,) owning row of each readout token
  ro_pos: Tensor|None = None   # (M_ro,) causal position of each readout token

@dataclass
class PrefixState:
  S: int                     # number of cached state tokens
  state_ids: tuple[int, ...] # token ids that produced this prefix state
  # per-layer state: DeltaNet -> (conv_state (3, qa) f16, recurrent_state (H, V, K) f16)
  #                  Attention -> (k_prefix (S, nkv, hd) f16, v_prefix (S, nkv*hd) f16)
  layers: tuple[tuple[Tensor, Tensor], ...]

  @property
  def tensors(self) -> tuple[Tensor, ...]:
    return tuple(t for pair in self.layers for t in pair)

def _ceil(x:int, a:int) -> int: return -(-x // a) * a

def layout_prefix(state_ids:list[int]) -> dict:
  """Host-side index tables for running the state prefix alone (`S` tokens) to build a `PrefixState`."""
  S = len(state_ids)
  assert S > 0, "prefix must have at least 1 token"
  L = _ceil(S, ROW_ALIGN)
  N = max(_ceil(S, ALIGN), 32)
  rows_idx = list(range(S)) + [S - 1] * (L - S)
  pack_idx = [t if t < S else 0 for t in range(N)]
  tok_row = [0] * N
  tok_pos = [t if t < S else 0 for t in range(N)]
  flags = [0] * N
  flags[S - 1] = 1
  conv_src = [[N, N, N] for _ in range(N)]
  for t in range(S):
    for j in range(3):
      if t >= 3 - j: conv_src[t][j] = t - 3 + j
  from .chunked_delta import make_chunks
  from .flash_attn import make_q_tiles
  chunk_tok, chunk_flags = make_chunks(S, [], N)
  q_tiles = make_q_tiles(S, [[]], L)
  return {"ids": list(state_ids) + [0] * (N - S), "S": S, "R": 1, "L": L, "N": N,
          "NC": len(chunk_tok), "NT": len(q_tiles), "rows_idx": rows_idx, "pack_idx": pack_idx,
          "flags": flags, "conv_src": conv_src, "chunk_tok": chunk_tok, "chunk_flags": chunk_flags,
          "tok_row": tok_row, "tok_pos": tok_pos, "q_tiles": q_tiles}

def layout_branches(enc:dict, S:int) -> dict:
  """Host-side index tables for branch-only packed inference given a cached `S`-token state prefix."""
  seg, n = enc["seg"], len(enc["ids"])
  assert seg.count(0) == S, f"prefix length mismatch: expected {S}, got {seg.count(0)}"
  n_b = n - S
  branches, start = [], S
  for d in enc["decide_idx"]:
    branches.append(list(range(start - S, d + 1 - S))); start = d + 1
  R = len(branches)
  L = _ceil(max(S + len(b) for b in branches), ROW_ALIGN)
  N_b = max(_ceil(n_b, ALIGN), 32)
  rows_idx, pack_idx, tok_row, tok_pos = [], [0] * N_b, [0] * N_b, [0] * N_b
  for r, b in enumerate(branches):
    # Attention KV table has [0..S-1] from prefix, [S..S+N_b-1] from branches
    row_kv = list(range(S)) + [S + pb for pb in b]
    for tb, pb in enumerate(b):
      pack_idx[pb] = r * L + S + tb
      tok_row[pb] = r
      tok_pos[pb] = S + tb
    rows_idx += row_kv + [row_kv[-1]] * (L - len(row_kv))
  decide = [d - S for d in enc["decide_idx"]]
  opts = [o - S for oi in enc["opt_idx"] for o in oi]
  owner = [q for q, oi in enumerate(enc["opt_idx"]) for _ in oi]
  flags = [0] * N_b
  for b in branches:
    if b: flags[b[0]] = 2
  # conv_src: 0..N_b-1 -> branch token, N_b..N_b+2 -> prefix conv_state[0..2], N_b+3 -> zero
  conv_src = [[N_b + 3, N_b + 3, N_b + 3] for _ in range(N_b)]
  for b in branches:
    for tb, pb in enumerate(b):
      for j in range(3):
        off = tb - 3 + j
        if off >= 0: conv_src[pb][j] = b[off]
        elif S + off >= 0: conv_src[pb][j] = N_b + 3 + off
  from .chunked_delta import make_chunks
  from .flash_attn import make_q_tiles
  chunk_tok, chunk_flags = make_chunks(0, branches, N_b)
  q_tiles = make_q_tiles(S, branches, L, prefix_cached=True)
  ro_raw = decide + opts
  M_ro = max(_ceil(len(ro_raw), ALIGN), ALIGN)
  ro_idx = ro_raw + [0] * (M_ro - len(ro_raw))
  ro_row = [tok_row[pb] for pb in ro_idx]
  ro_pos = [tok_pos[pb] for pb in ro_idx]
  return {"ids": enc["ids"][S:] + [0] * (N_b - n_b), "S": S, "R": R, "L": L, "N": N_b,
          "NC": len(chunk_tok), "NT": len(q_tiles), "M_ro": M_ro,
          "rows_idx": rows_idx, "pack_idx": pack_idx, "flags": flags, "conv_src": conv_src,
          "chunk_tok": chunk_tok, "chunk_flags": chunk_flags, "tok_row": tok_row, "tok_pos": tok_pos,
          "q_tiles": q_tiles, "ro_idx": ro_idx, "ro_row": ro_row, "ro_pos": ro_pos,
          "decide": decide, "opts": opts, "owner": owner}

def layout(enc:dict) -> dict:
  """Host-side index tables for one packed encoding (lists, turned into Tensors by the caller)."""
  seg, n = enc["seg"], len(enc["ids"])
  S = seg.count(0)
  branches, start = [], S
  for d in enc["decide_idx"]:
    branches.append(list(range(start, d + 1))); start = d + 1
  R = len(branches)
  L = _ceil(max(S + len(b) for b in branches), ROW_ALIGN)
  N = max(_ceil(n, ALIGN), 32)
  rows_idx, pack_idx, tok_row, tok_pos = [], [0] * N, [0] * N, [0] * N
  for r, b in enumerate(branches):
    row = list(range(S)) + b
    for t, p in enumerate(row):
      if r == 0 or p >= S:
        pack_idx[p] = r * L + t
        tok_row[p] = r
        tok_pos[p] = t
    rows_idx += row + [row[-1]] * (L - len(row))
  opts = [o for oi in enc["opt_idx"] for o in oi]
  owner = [q for q, oi in enumerate(enc["opt_idx"]) for _ in oi]
  # the scan walks the packed order once: the prefix's last token saves the state, every branch start restores it
  flags = [0] * N
  if S: flags[S-1] = 1
  start = S
  for d in enc["decide_idx"]:
    flags[start] = 2; start = d + 1
  # the packed conv reads the three tokens before each token inside its own row; a fresh row starts from zeros
  conv_src = [[N, N, N] for _ in range(N)]
  for b, end in zip(branches, enc["decide_idx"]):
    row = list(range(S)) + b
    for t, p in enumerate(row):
      for j in range(3):
        if t >= 3 - j: conv_src[p][j] = row[t - 3 + j]
  from .chunked_delta import make_chunks
  from .flash_attn import make_q_tiles
  chunk_tok, chunk_flags = make_chunks(S, branches, N)
  q_tiles = make_q_tiles(S, branches, L)
  decide = list(enc["decide_idx"])
  ro_raw = decide + opts
  M_ro = max(_ceil(len(ro_raw), ALIGN), ALIGN)
  ro_idx = ro_raw + [0] * (M_ro - len(ro_raw))
  ro_row = [tok_row[p] for p in ro_idx]
  ro_pos = [tok_pos[p] for p in ro_idx]
  return {"ids": enc["ids"] + [0] * (N - n), "R": R, "L": L, "N": N, "NC": len(chunk_tok), "NT": len(q_tiles),
          "M_ro": M_ro, "rows_idx": rows_idx, "pack_idx": pack_idx, "flags": flags, "conv_src": conv_src,
          "chunk_tok": chunk_tok, "chunk_flags": chunk_flags, "tok_row": tok_row, "tok_pos": tok_pos,
          "q_tiles": q_tiles, "ro_idx": ro_idx, "ro_row": ro_row, "ro_pos": ro_pos,
          "decide": decide, "opts": opts, "owner": owner}

def to_rows(x:Tensor, pk:Pack) -> Tensor:
  """(N, C) packed -> (R, L, C) rows."""
  return gather_rows(x.reshape(pk.N, -1), pk.rows_idx).reshape(pk.R, pk.L, -1)

def to_packed(x:Tensor, pk:Pack) -> Tensor:
  """(R, L, C) rows -> (N, C) packed."""
  return gather_rows(x.reshape(pk.R * pk.L, -1), pk.pack_idx)

def _out_proj(lin, x:Tensor, splitk:bool) -> Tensor:
  """Apply output Linear; when `splitk=True` and supported, returns `(SK, N, D)` partials for `add_rmsnorm`."""
  if splitk and lin.bias is None and getattr(lin, "groups", 1) == 1 and (wt := getattr(lin, "_weight_t", None)) is not None:
    from .gemm import matmul_t_splitk
    if (parts := matmul_t_splitk(x, wt)) is not None: return parts
  return lin(x)

def delta_attention(blk, x:Tensor, pk:Pack, splitk:bool=False,
                    conv_prefix:Tensor|None=None, init_state:Tensor|None=None,
                    return_state:int=0) -> Tensor|tuple[Tensor, Tensor, Tensor]:
  """`GatedDeltaNetBlock._attention` (non-KDA) on a packed sequence."""
  from . import conv as C
  cfg = blk.config.ssm
  N, R, L = pk.N, pk.R, pk.L
  H, V, K = blk.num_v_heads, blk.head_v_dim, blk.head_k_dim
  xh = x.half()
  p = blk._proj(xh)                                                   # qkv | gate | beta | alpha, one GEMM
  qa, qb, qc = blk._proj_ends
  from .chunked_delta import fused_delta_core
  saved_state = None
  res = fused_delta_core(blk, p, pk.chunk_tok, pk.chunk_flags, pk.conv_src,
                         conv_prefix=conv_prefix, init_state=init_state, return_state=return_state > 0)
  if res is not None:
    if return_state > 0: core, saved_state = res
    else: core = res
  else:
    if conv_prefix is not None or init_state is not None or return_state > 0:
      raise RuntimeError("cached prefix requires Metal fp16 chunked DeltaNet")
    beta = p[..., qb:qc].float().sigmoid().reshape(N, H)
    g = (p[..., qc:].float() + blk.ssm_dt["bias"]).reshape(N, H)
    a = blk.ssm_a.reshape(H)
    log_alpha = g.softplus() * a if cfg.gate_lower_bound is None else (g * a.abs()).sigmoid() * cfg.gate_lower_bound
    proj = p[..., :qa].reshape(N, -1)                                 # (N, C) f16
    conv = _conv(blk, proj, pk, C)                                    # (N, C) f32
    q, k, v = conv.split([blk.q_dim, blk.q_dim, blk.conv_channels - 2*blk.q_dim], dim=-1)
    q, k = (z.reshape(N, blk.num_k_heads, K) for z in (q, k))
    q, k = ((z.rmsnorm(eps=1e-6/K) / K**0.5).repeat(1, H // blk.num_k_heads, 1) for z in (q, k))
    v = v.reshape(N, H, V)
    q, k, v = (z.transpose(0, 1).float() for z in (q, k, v))          # (H, N, K|V)
    q = q * K**-0.5
    core = _delta_core(q, k, v, beta.transpose(0, 1), log_alpha.transpose(0, 1).exp().reshape(H, N, 1), pk)
  from .rmsnorm import gated_rmsnorm
  if (z := gated_rmsnorm(core, p, blk.ssm_norm.weight, qa, blk.config.norm_eps, cfg.silu_gate)) is not None:
    out = _out_proj(blk.ssm_out, z.reshape(1, N, -1), splitk)
  else:
    out_gate = p[..., qa:qb].reshape(N, H, V)
    core = core.transpose(0, 1)                                       # (N, H, V)
    z = (blk.ssm_norm(core) * (out_gate.silu() if cfg.silu_gate else out_gate.sigmoid())).cast(x.dtype)
    out = _out_proj(blk.ssm_out, z.reshape(1, N, -1), splitk)
  if return_state > 0:
    p2 = p.reshape(N, -1)
    S = return_state
    if S >= 3: conv_st = p2[S - 3 : S, :qa].contiguous()
    else: conv_st = Tensor.zeros(3 - S, qa, dtype=p2.dtype, device=p2.device).cat(p2[:S, :qa], dim=0).contiguous()
    return out, conv_st, saved_state
  return out

def _conv(blk, proj:Tensor, pk:Pack, C) -> Tensor:
  """The 4-tap causal conv, in packed order but with each token's window taken from its own row."""
  w = blk.ssm_conv1d["weight"].flatten(1)
  if (z := C.short_conv_packed(proj, w, pk.conv_src)) is not None: return z
  conv_rows, _ = C.short_conv(Tensor.zeros(pk.R, w.shape[1] - 1, proj.shape[-1], device=proj.device), to_rows(proj, pk),
                              w, Tensor(True, device=proj.device))
  return to_packed(conv_rows, pk)

def _delta_core(q:Tensor, k:Tensor, v:Tensor, bt:Tensor, alpha_in:Tensor, pk:Pack) -> Tensor:
  """`(H, N, K)`/`(H, N, V)` packed scan operands -> `(H, N, V)` scan output."""
  from .gated_delta import ROW_TILE, _is_metal, get_gated_delta_packed, packed_kernel_params
  H, N, K = q.shape
  V = v.shape[-1]
  if K % 32 or V % ROW_TILE or not _is_metal(q.device): return _scan_ref(q, k, v, bt, alpha_in, pk.flags)
  kq = (q * k).sum(-1).contiguous()
  kern = get_gated_delta_packed().specialize(**packed_kernel_params(heads=H, tokens=N, k_blks=K // 32,
                                                                   v_tiles=V // 4, alpha_k=1))
  out = UOp.alloc((H, N, V), dtypes.float32, device=q.device)
  return Tensor(out.after(kern.call(out, q.contiguous().uop, k.contiguous().uop, v.contiguous().uop,
                                    bt.contiguous().uop, alpha_in.contiguous().uop, kq.uop, pk.flags.uop)))

def _scan_ref(q, k, v, beta, alpha, flags:Tensor) -> Tensor:
  """The recurrence in Tensor ops over the packed order (tiny test models whose head dims the UIR scan does not tile)."""
  H, N, K = q.shape
  state = saved = Tensor.zeros(H, v.shape[-1], K, device=q.device)
  outs = []
  for t in range(N):
    state = flags[t].eq(2).where(saved, state)
    s1 = state * alpha[:, t].reshape(H, 1, 1)
    delta = (v[:, t].unsqueeze(-1) - (s1 * k[:, t].unsqueeze(-2)).sum(-1, keepdim=True)) * beta[:, t].reshape(H, 1, 1)
    state = s1 + delta * k[:, t].unsqueeze(-2)
    saved = flags[t].eq(1).where(state, saved)
    outs.append((state * q[:, t].unsqueeze(-2)).sum(-1))
  return outs[0].stack(*outs[1:], dim=1)

def attention(blk, x:Tensor, pk:Pack, splitk:bool=False,
              k_prefix:Tensor|None=None, v_prefix:Tensor|None=None,
              return_kv:int=0) -> Tensor|tuple[Tensor, Tensor, Tensor]:
  """`TransformerBlock._attention` on a packed sequence (each row is causal from position 0)."""
  cfg = blk.config
  N, R, L, nh, nkv, hd = pk.N, pk.R, pk.L, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
  if (qkv_lin := getattr(blk, "_qkv", None)) is not None:
    a, b = blk._qkv_ends
    p = qkv_lin(x)
    from .flash_attn import packed_attention
    res = packed_attention(blk, p, pk, k_prefix=k_prefix, v_prefix=v_prefix, return_kv=return_kv)
    if res is not None:
      if return_kv > 0:
        out, k_st, v_st = res
        return _out_proj(blk.attn_output, out.reshape(1, N, nh * hd), splitk), k_st, v_st
      return _out_proj(blk.attn_output, res.reshape(1, N, nh * hd), splitk)
    if k_prefix is not None or return_kv > 0:
      raise RuntimeError("cached prefix requires Metal fp16 packed Flash Attention")
    q, k, v = p[..., :a], p[..., a:b], p[..., b:]
  else:
    q, k, v = blk.attn_q(x), blk.attn_k(x), blk.attn_v(x)
  if cfg.qk_norm and cfg.qk_norm != hd: q, k = blk.attn_q_norm(q), blk.attn_k_norm(k)
  gate = None
  if cfg.attn_output_gate:
    qg = q.reshape(N, nh, 2, hd)
    q, gate = qg[:, :, 0, :], qg[:, :, 1, :].reshape(N, nh * hd)
  q, k, v = q.reshape(N, nh, hd), k.reshape(N, nkv, hd), v.reshape(N, nkv, hd)
  if cfg.qk_norm == hd: q, k = blk.attn_q_norm(q), blk.attn_k_norm(k)
  q, k, v = (to_rows(z.reshape(N, -1), pk).reshape(R, L, -1, hd).transpose(1, 2) for z in (q, k, v))
  freqs = blk.freqs_cis[0:L]
  q = apply_rope(q[..., :cfg.rope_dim], freqs).cat(q[..., cfg.rope_dim:], dim=-1)
  k = apply_rope(k[..., :cfg.rope_dim], freqs).cat(k[..., cfg.rope_dim:], dim=-1)
  attn = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True)      # (R, H, L, hd)
  attn = to_packed(attn.transpose(1, 2).cast(x.dtype), pk).reshape(1, N, nh * hd)
  return _out_proj(blk.attn_output, attn if gate is None else attn * gate.reshape(1, N, -1).sigmoid(), splitk)

def feed_forward(blk, x:Tensor, tokens:Tensor|None=None, splitk:bool=False) -> Tensor|None:
  """Fused SwiGLU MLP fast path when the block uses standard SiLU SwiGLU with pre-transposed f16 weights."""
  cfg = blk.config
  if cfg.num_experts or cfg.situ is not None or cfg.swiglu_limit is not None or cfg.swiglu_alpha != 1.0 or cfg.swiglu_up_bias != 0.0:
    return None
  wgt, wut = getattr(blk.ffn_gate, "_weight_t", None), getattr(blk.ffn_up, "_weight_t", None)
  if wgt is None or wut is None: return None
  from .gemm import swiglu_matmul_t
  if (mid := swiglu_matmul_t(x, wgt, wut)) is None: return None
  return _out_proj(blk.ffn_down, mid, splitk)

class PackedTransformerBlock(TransformerBlock):
  def _attention(self, x, start_pos, idx_mask=None):
    if (pk := getattr(self, "_pack", None)) is not None: return attention(self, x, pk)
    return super()._attention(x, start_pos, idx_mask)
  def _feed_forward(self, x, tokens=None):
    if (out := feed_forward(self, x, tokens)) is not None: return out
    return super()._feed_forward(x, tokens)

def install(model) -> None:
  from tinygrad.llm.model import GatedDeltaNetBlock
  from .gated_delta import PortableGatedDeltaNetBlock
  for blk in model.blk:
    if type(blk) is TransformerBlock: blk.__class__ = PackedTransformerBlock
    elif type(blk) is GatedDeltaNetBlock: blk.__class__ = PortableGatedDeltaNetBlock

def _attention_readout(blk, h_all:Tensor, h_ro:Tensor, pk:Pack, splitk:bool,
                       k_prefix:Tensor|None=None, v_prefix:Tensor|None=None) -> Tensor|None:
  """Exact lossless final-layer attention: K/V on all `N` tokens, Q/attention/output only on `M_ro` readout tokens."""
  kv_t, qt = getattr(blk, "_kv_t", None), getattr(blk.attn_q, "_weight_t", None)
  if kv_t is None or qt is None or pk.ro_pos is None or pk.ro_row is None: return None
  from .flash_attn import packed_attention_readout
  from .gemm import matmul_t
  if (kv_all := matmul_t(h_all, kv_t)) is None: return None
  if (q_ro := matmul_t(h_ro, qt)) is None: return None
  if (out_ro := packed_attention_readout(blk, q_ro, kv_all, pk, k_prefix=k_prefix, v_prefix=v_prefix)) is None: return None
  return _out_proj(blk.attn_output, out_ro.reshape(1, h_ro.shape[0], -1), splitk)

def forward_prefix(model, ids:Tensor, pk:Pack, S:int) -> tuple[tuple[Tensor, Tensor], ...]:
  """Run the backbone on a shared state prefix (`S` tokens) and capture per-layer DeltaNet + KV cache states."""
  from tinygrad.llm.model import GatedDeltaNetBlock
  from .rmsnorm import add_rmsnorm, rmsnorm
  eps = model.config.norm_eps
  x = model.token_embd(ids)[0]
  blks = model.blk
  h = rmsnorm(x, blks[0].attn_norm.weight, eps)
  states: list[tuple[Tensor, Tensor]] = []
  last_i = len(blks) - 1
  for i, blk in enumerate(blks):
    if isinstance(blk, TransformerBlock):
      if not hasattr(blk, "freqs_cis"): blk._init_state(h.unsqueeze(0))
      attn_out, k_st, v_st = attention(blk, h.unsqueeze(0), pk, splitk=True, return_kv=S)
      states.append((k_st, v_st))
    elif isinstance(blk, GatedDeltaNetBlock) and not blk.config.ssm.kda:
      attn_out, conv_st, rec_st = delta_attention(blk, h.unsqueeze(0), pk, splitk=True, return_state=S)
      states.append((conv_st, rec_st))
    else:
      raise RuntimeError(f"cached prefix does not support block {type(blk).__name__}")
    if i == last_i: break   # final layer only needs its KV/recurrent state; output hidden states aren't read
    if attn_out.ndim == 3 and attn_out.shape[0] == 1: attn_out = attn_out[0]
    x, h = add_rmsnorm(x, attn_out, blk.ffn_norm.weight, eps)
    if (ffn_out := feed_forward(blk, h.unsqueeze(0), splitk=True)) is None: ffn_out = blk._feed_forward(h.unsqueeze(0), ids)
    if ffn_out.ndim == 3 and ffn_out.shape[0] == 1: ffn_out = ffn_out[0]
    x, h = add_rmsnorm(x, ffn_out, blks[i + 1].attn_norm.weight, eps)
  return tuple(states)

def forward_packed(model, ids:Tensor, pk:Pack, readout_only:bool=False,
                   prefix_layers:tuple[tuple[Tensor, Tensor], ...]|None=None) -> tuple[Tensor, bool]:
  """Walk the backbone in packed order `(N, D)` fusing every residual add (and Split-K reduction) with the next RMSNorm.

  When `readout_only=True` and the final block is a fused full-attention block with `M_ro < N`, prunes the
  final layer to compute only the `M_ro` readout tokens (`decide + opts`) — 100% lossless.  Returns `(h, pruned)`."""
  from tinygrad.llm.model import GatedDeltaNetBlock
  from .rmsnorm import add_rmsnorm, rmsnorm
  eps = model.config.norm_eps
  x = model.token_embd(ids)[0]                                        # (N, D)
  blks = model.blk
  h = rmsnorm(x, blks[0].attn_norm.weight, eps)
  last_i = len(blks) - 1
  for i, blk in enumerate(blks):
    pst = prefix_layers[i] if prefix_layers is not None else None
    if i == last_i and readout_only and pk.ro_idx is not None and pk.ro_idx.shape[0] < pk.N \
        and isinstance(blk, TransformerBlock) and hasattr(blk, "_kv_t"):
      if not hasattr(blk, "freqs_cis"): blk._init_state(h.unsqueeze(0))
      x_ro, h_ro = gather_rows(x, pk.ro_idx), gather_rows(h, pk.ro_idx)
      kp, vp = pst if pst is not None else (None, None)
      if (attn_ro := _attention_readout(blk, h, h_ro, pk, splitk=True, k_prefix=kp, v_prefix=vp)) is not None:
        if attn_ro.ndim == 3 and attn_ro.shape[0] == 1: attn_ro = attn_ro[0]
        x_ro, h_ro = add_rmsnorm(x_ro, attn_ro, blk.ffn_norm.weight, eps)
        if (ffn_ro := feed_forward(blk, h_ro.unsqueeze(0), splitk=True)) is None: ffn_ro = blk._feed_forward(h_ro.unsqueeze(0), ids)
        if ffn_ro.ndim == 3 and ffn_ro.shape[0] == 1: ffn_ro = ffn_ro[0]
        _, h_ro = add_rmsnorm(x_ro, ffn_ro, model.output_norm.weight, eps)
        return h_ro, True
    if isinstance(blk, TransformerBlock):
      if not hasattr(blk, "freqs_cis"): blk._init_state(h.unsqueeze(0))
      kp, vp = pst if pst is not None else (None, None)
      attn_out = attention(blk, h.unsqueeze(0), pk, splitk=True, k_prefix=kp, v_prefix=vp)
    elif isinstance(blk, GatedDeltaNetBlock) and not blk.config.ssm.kda:
      cp, sp = pst if pst is not None else (None, None)
      attn_out = delta_attention(blk, h.unsqueeze(0), pk, splitk=True, conv_prefix=cp, init_state=sp)
    else:
      # fallback to generic block call for unsupported layer types
      set_pack(model, pk)
      try: x = blk(x.unsqueeze(0), (0,), None, ids, None)[0][0]
      finally: set_pack(model, None)
      nxt_w = blks[i + 1].attn_norm.weight if i + 1 < len(blks) else model.output_norm.weight
      h = rmsnorm(x, nxt_w, eps)
      continue
    if attn_out.ndim == 3 and attn_out.shape[0] == 1: attn_out = attn_out[0]
    x, h = add_rmsnorm(x, attn_out, blk.ffn_norm.weight, eps)
    if (ffn_out := feed_forward(blk, h.unsqueeze(0), splitk=True)) is None: ffn_out = blk._feed_forward(h.unsqueeze(0), ids)
    if ffn_out.ndim == 3 and ffn_out.shape[0] == 1: ffn_out = ffn_out[0]
    nxt_w = blks[i + 1].attn_norm.weight if i + 1 < len(blks) else model.output_norm.weight
    x, h = add_rmsnorm(x, ffn_out, nxt_w, eps)
  return h, False

def set_pack(model, pk:Pack|None) -> None:
  for blk in model.blk: blk._pack = pk
