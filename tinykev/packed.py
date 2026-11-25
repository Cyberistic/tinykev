"""Packed inference: every token once, exactly Kev's packed encoding.

The row form recomputes the state for every question and pads every row to the longest one (Kev-0.8B bench
request: rows of 64/37/44 tokens -> 192 GEMM rows for 101 real tokens).  Here the token-wise work -- embedding,
norms, every projection and the MLPs, ~80 % of the GPU time -- runs on the packed sequence (state once, then each
branch), and only the sequence mixers see rows:

* Gated DeltaNet: the qkv projection is gathered into `state + branch` rows, the short conv and the scan run on
  the rows exactly as before, and the scan output is gathered back to packed order before the gated norm.
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

def _ceil(x:int, a:int) -> int: return -(-x // a) * a

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
  rows_idx, pack_idx = [], [0] * N
  for r, b in enumerate(branches):
    row = list(range(S)) + b
    for t, p in enumerate(row):
      if r == 0 or p >= S: pack_idx[p] = r * L + t
    rows_idx += row + [row[-1]] * (L - len(row))
  opts = [o for oi in enc["opt_idx"] for o in oi]
  owner = [q for q, oi in enumerate(enc["opt_idx"]) for _ in oi]
  return {"ids": enc["ids"] + [0] * (N - n), "R": R, "L": L, "N": N, "rows_idx": rows_idx, "pack_idx": pack_idx,
          "decide": list(enc["decide_idx"]), "opts": opts, "owner": owner}

def to_rows(x:Tensor, pk:Pack) -> Tensor:
  """(N, C) packed -> (R, L, C) rows."""
  return gather_rows(x.reshape(pk.N, -1), pk.rows_idx).reshape(pk.R, pk.L, -1)

def to_packed(x:Tensor, pk:Pack) -> Tensor:
  """(R, L, C) rows -> (N, C) packed."""
  return gather_rows(x.reshape(pk.R * pk.L, -1), pk.pack_idx)

_zero_state:dict[tuple, Tensor] = {}
def _zeros(shape:tuple, device) -> Tensor:
  """A realized zero buffer (the scan's unused initial-state operand); `prepare` creates it outside the graph."""
  key = (shape, device)
  if key not in _zero_state: _zero_state[key] = Tensor.zeros(*shape, dtype=dtypes.float32, device=device).contiguous().realize()
  return _zero_state[key]

def prepare(model, R:int, device) -> None:
  from tinygrad.llm.model import GatedDeltaNetBlock
  for blk in model.blk:
    if isinstance(blk, GatedDeltaNetBlock): _zeros((R, blk.num_v_heads, blk.head_v_dim, blk.head_k_dim), device)

def delta_attention(blk, x:Tensor, pk:Pack) -> Tensor:
  """`GatedDeltaNetBlock._attention` (non-KDA) on a packed sequence."""
  from . import conv as C
  from .gated_delta import ROW_TILE, _is_metal
  cfg = blk.config.ssm
  N, R, L = pk.N, pk.R, pk.L
  H, V, K = blk.num_v_heads, blk.head_v_dim, blk.head_k_dim
  xh = x.half()
  out_gate = blk.attn_gate(xh).reshape(N, H, V)
  beta = blk.ssm_beta(xh).float().sigmoid().reshape(N, H)
  g = (blk.ssm_alpha(xh).float() + blk.ssm_dt["bias"]).reshape(N, H)
  a = blk.ssm_a.reshape(H)
  log_alpha = g.softplus() * a if cfg.gate_lower_bound is None else (g * a.abs()).sigmoid() * cfg.gate_lower_bound
  ba = to_rows(beta.cat(log_alpha, dim=-1), pk)                       # (R, L, 2H) f32
  proj = to_rows(blk.attn_qkv(xh).reshape(N, -1), pk)                 # (R, L, C) f16
  w = blk.ssm_conv1d["weight"].flatten(1)
  conv_out, _ = C.short_conv(Tensor.zeros(R, w.shape[1] - 1, proj.shape[-1], device=proj.device), proj, w,
                             Tensor(True, device=proj.device))
  q, k, v = conv_out.split([blk.q_dim, blk.q_dim, blk.conv_channels - 2*blk.q_dim], dim=-1)
  q, k = (z.reshape(R, L, blk.num_k_heads, K) for z in (q, k))
  q, k = ((z.rmsnorm(eps=1e-6/K) / K**0.5).repeat(1, 1, H // blk.num_k_heads, 1) for z in (q, k))
  v = v.reshape(R, L, H, V)
  q, k, v = (z.transpose(1, 2).float() for z in (q, k, v))
  q = q * K**-0.5
  bt = ba[..., :H].transpose(1, 2)
  alpha_in = ba[..., H:].transpose(1, 2).exp().reshape(R, H, L, 1)
  if K % 32 or V % ROW_TILE or not _is_metal(proj.device):
    core = _scan_ref(q, k, v, bt, alpha_in)
  else: core = _scan_uir(q, k, v, bt, alpha_in, R, H, L, V, K, proj.device)
  core = to_packed(core.transpose(1, 2), pk).reshape(N, H, V)
  z = (blk.ssm_norm(core) * (out_gate.silu() if cfg.silu_gate else out_gate.sigmoid())).cast(x.dtype)
  return blk.ssm_out(z.reshape(1, N, -1))

def _scan_ref(q, k, v, beta, alpha) -> Tensor:
  """The recurrence in Tensor ops (tiny test models whose head dims the UIR scan does not tile)."""
  R, H, L, K = q.shape
  state = Tensor.zeros(R, H, v.shape[-1], K, device=q.device)
  outs = []
  for t in range(L):
    s1 = state * alpha[:, :, t].reshape(R, H, 1, 1)
    delta = (v[:, :, t].unsqueeze(-1) - (s1 * k[:, :, t].unsqueeze(-2)).sum(-1, keepdim=True)) * beta[:, :, t].reshape(R, H, 1, 1)
    state = s1 + delta * k[:, :, t].unsqueeze(-2)
    outs.append((state * q[:, :, t].unsqueeze(-2)).sum(-1))
  return outs[0].stack(*outs[1:], dim=2)

def _scan_uir(q, k, v, bt, alpha_in, R, H, L, V, K, device) -> Tensor:
  from .gated_delta import get_gated_delta, kernel_params
  kq = (q * k).sum(-1).contiguous()
  kern = get_gated_delta().specialize(**kernel_params(batch=R, heads=H, tokens=L, v_tiles=V // 4, k_blks=K // 32, alpha_k=1))
  out = UOp.alloc((R, H, L, V), dtypes.float32, device=device)
  return Tensor(out.after(kern.call(out, q.contiguous().uop, k.contiguous().uop, v.contiguous().uop, bt.contiguous().uop,
                                    alpha_in.contiguous().uop, _zeros((R, H, V, K), device).uop, kq.uop,
                                    UOp.const(0, dtypes.weakint))))

def attention(blk, x:Tensor, pk:Pack) -> Tensor:
  """`TransformerBlock._attention` on a packed sequence (each row is causal from position 0)."""
  cfg = blk.config
  N, R, L, nh, nkv, hd = pk.N, pk.R, pk.L, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
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
  return blk.attn_output(attn if gate is None else attn * gate.reshape(1, N, -1).sigmoid())

class PackedTransformerBlock(TransformerBlock):
  def _attention(self, x, start_pos, idx_mask=None):
    if (pk := getattr(self, "_pack", None)) is not None: return attention(self, x, pk)
    return super()._attention(x, start_pos, idx_mask)

def install(model) -> None:
  from tinygrad.llm.model import GatedDeltaNetBlock
  from .gated_delta import PortableGatedDeltaNetBlock
  for blk in model.blk:
    if type(blk) is TransformerBlock: blk.__class__ = PackedTransformerBlock
    elif type(blk) is GatedDeltaNetBlock: blk.__class__ = PortableGatedDeltaNetBlock

def set_pack(model, pk:Pack|None) -> None:
  for blk in model.blk: blk._pack = pk
