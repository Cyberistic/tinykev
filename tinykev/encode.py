"""Kev's token layout, ported from `kev.model.encode` / `rows_of`, plus the `/v1/systemone` request mapping
from `kev.api`.

Kev reads a document and a set of questions with one causal LM and a pointer head.  The LM sees

    [<state>] state tokens ... [<q>] instruction [<opt>] option0 [/opt] [<opt>] option1 [/opt] ... [<decide>]

per question, and the head scores `<decide>` against the `</opt>` token of every option.  Questions may not
read each other, so the packed attention is block-causal: state is visible to everything, a question sees the
state and itself.  Qwen3.5/3.8 have Gated DeltaNet layers, whose recurrent state cannot honour an arbitrary
mask, so Kev always runs the equivalent *row* form: one causal row per question, `state + branch`.  The rows
are independent, which is what makes the port simple.
"""
from __future__ import annotations

import json, re
from dataclasses import dataclass

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
SERVE_MAX_STATE, SERVE_MAX_BRANCH = 65536, 65536 + 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH
OPT_NONE, OPT_DECIDE = -1, -2

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

class ContextOverflow(ValueError):
  def __init__(self, message, state_tokens=None, max_state=None):
    super().__init__(message)
    self.state_tokens, self.max_state = state_tokens, max_state

def user_tokens(tok, text:str) -> list[int]:
  """Caller text can never produce delimiter/control tokens: `<|name|>` is rewritten to `<¦name¦>` before
  tokenizing, so an option boundary cannot be forged by the document."""
  return tok.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False)

def encode(tok, rec:dict, max_state:int=MAX_STATE, max_branch:int=MAX_BRANCH, strict:bool=False) -> dict:
  """Pack one record into ids / seg / pos / decide_idx / opt_idx, exactly like `kev.model.encode`.

  seg: 0 for state, k for question k.  pos: branch positions restart after the state.  opt: per-token option
  index (OPT_NONE for state/instruction, 0..K-1 inside an option span, OPT_DECIDE for <decide>)."""
  state_tokens = user_tokens(tok, rec["state"])
  if strict and len(state_tokens) + 1 > max_state:
    raise ContextOverflow(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}", state_tokens=len(state_tokens) + 1, max_state=max_state)
  S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[:max_state-1]
  ids, seg, pos, opt = list(S), [0]*len(S), list(range(len(S))), [OPT_NONE]*len(S)
  q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
  decide_idx, opt_idx = [], []
  for k, q in enumerate(rec["questions"], start=1):
    instr = [q_id] + user_tokens(tok, q["instr"])
    spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
    br = instr + [t for sp in spans for t in sp] + [d_id]
    if len(br) > max_branch - len(S):
      raise ContextOverflow(f"branch too long: {len(br)} tokens with a {len(S)}-token state (row limit {max_branch})")
    base = len(ids)
    br_opt = [OPT_NONE]*len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
    ends, cursor = [], len(instr)
    for sp in spans:
      cursor += len(sp); ends.append(cursor - 1)
    ids += br; seg += [k]*len(br); pos += list(range(len(S), len(S)+len(br))); opt += br_opt
    decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
  return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "decide_idx": decide_idx, "opt_idx": opt_idx,
          "labels": [q["label"] for q in rec["questions"]], "state_tokens": len(state_tokens)+1,
          "state_truncated": len(state_tokens)+1 > max_state}

def rows_of(enc:dict) -> tuple[list[int], list[int], list[dict]]:
  """Split a packed encoding into `(state_ids, state_pos, rows)` where each row is one question's branch.

  `state + row` is a plain causal sequence with positions 0..L-1, so a row is a single prefill call."""
  seg, Ls = enc["seg"], enc["seg"].count(0)
  rows, start = [], Ls
  for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
    end = d + 1
    if seg[start] != k or seg[end-1] != k: raise ValueError("branch layout mismatch")
    rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": d-start, "opts": [o-start for o in oi]})
    start = end
  return enc["ids"][:Ls], enc["pos"][:Ls], rows

# ---- /v1/systemone request mapping (kev.api) ----

MAX_OPTIONS = 255

def render(v, indent:int=0) -> str:
  pad = "  " * indent
  if v is None: return ""
  if isinstance(v, (str, int, float, bool)): return str(v)
  if isinstance(v, list): return "\n".join(f"{pad}- {x if isinstance(x, str) else render(x, indent+1).lstrip()}" for x in v)
  return "\n".join(f"{pad}{k}:\n{render(x, indent+1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())

def option_text(name:str, desc) -> str:
  return name if desc is None or desc == "" else f"{name}: {render(desc)}"

def question_keys(qtype:str, criteria) -> list[str]:
  if qtype == "choice": return list(criteria)
  if qtype == "noul": return ["false", "true"]
  return [str(i) for i in range(len(criteria))]

def to_record(req:dict) -> tuple[dict, list[dict]]:
  qs, meta = [], []
  for qid, q in req["questions"].items():
    m = {"id": qid, "type": q["type"], "keys": question_keys(q["type"], q.get("criteria"))}
    if q["type"] == "noul":
      c = q.get("criteria") or {}
      opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
    elif q["type"] == "choice":
      opts = [option_text(k, v) for k, v in q["criteria"].items()]
    else:
      opts = [render(x) for x in q["criteria"]]
      m["legend"] = dict(zip(m["keys"], opts))
    qs.append({"instr": render(q.get("instructions")), "options": opts, "label": 0}); meta.append(m)
  return {"state": render(req["state"]), "questions": qs}, meta

def _normalize(p:list[float]) -> list[float]:
  t = sum(p)
  return [1/len(p)]*len(p) if t == 0 else [x/t for x in p]

def choice_confidence(p:list[float]) -> float:
  K = len(p)
  return 1.0 if K == 1 else (max(_normalize(p)) - 1/K) / (1 - 1/K)

def score_confidence(p:list[float]) -> float:
  L = len(p)
  if L == 1: return 1.0
  p = _normalize(p); mode = max(range(L), key=p.__getitem__)
  D = sum(abs(i - (L-1)/2) for i in range(L)) / L
  return max(0.0, 1.0 - sum(pi*abs(i-mode) for i, pi in enumerate(p)) / D)

def round_prob(x:float) -> float: return round(float(x), 4)

def to_answers(probs:list[list[float]], meta:list[dict]) -> dict:
  out = {}
  for p, m in zip(probs, meta):
    if m["type"] == "noul":
      out[m["id"]] = {"type": "noul", "noul": round_prob(p[1])}
    elif m["type"] == "choice":
      out[m["id"]] = {"type": "choice", "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                      "confidence": round_prob(choice_confidence(p)), "probabilities": {k: round_prob(v) for k, v in zip(m["keys"], p)}}
    else:
      out[m["id"]] = {"type": "score", "score": round_prob(sum(i*pi for i, pi in enumerate(p))), "legend": m["legend"],
                      "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)}, "confidence": round_prob(score_confidence(p))}
  return out
