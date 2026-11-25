"""tinykev: run Kev decision models on tinygrad.

```python
from tinykev import load_kev, encode, to_record, to_answers

tok, kev, meta = load_kev("jaredpalmer/kev-0.8b")
rec, meta_q = to_record(request)
enc = encode(tok, rec, max_state=8192, max_branch=8192)
print(to_answers(kev.probs(enc), meta_q))
```
"""
from __future__ import annotations

import json
from pathlib import Path

from .encode import ContextOverflow, encode, rows_of, to_answers, to_record
from .model import Kev
from .tokenizer import Tokenizer
from .weights import load, read_meta, resolve

__all__ = ["load_kev", "encode", "rows_of", "to_record", "to_answers", "ContextOverflow", "Kev", "Tokenizer"]

def load_kev(run:str, *, max_context:int=8192, device:str|None=None, dtype=None, jit:bool=False) -> tuple[Tokenizer, Kev, dict]:
  """Load a Kev checkpoint (`jaredpalmer/kev-0.8b`, `...@revision`, or a local run directory).

  `max_context` sizes the KV cache allocated by the attention layers; it must be at least the longest
  `state + branch` row.  It is not the model's rotary limit (262144 for the Qwen bases)."""
  kw = {} if dtype is None else {"dtype": dtype}
  loaded = load(run, max_context=max_context, device=device, **kw)
  tok = Tokenizer(loaded.model_dir / "tokenizer.json")
  return tok, Kev(loaded, device=device, use_jit=jit), loaded.meta
