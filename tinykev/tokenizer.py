"""Byte-level BPE tokenizer read straight from the `tokenizer.json` a Kev checkpoint ships.

This is a dependency-free port of the pieces of HF `tokenizers` that Qwen's tokenizer actually uses: a
`Split` regex pre-tokenizer, `ByteLevel` bytes<->unicode mapping and the BPE merge table, plus NFC
normalization and the added/special token trie.  It is validated against `tokenizers` in
`tinykev/test/test_tokenizer.py` when that package is installed.

Kev's encoder needs only `encode(text, add_special_tokens=False)`; `convert_tokens_to_ids` is here for
the delimiter tokens (`<|fim_prefix|>`, ...) that Kev reuses for state/question/option markers.
"""
from __future__ import annotations

import json, re, unicodedata
from pathlib import Path

# ---- unicode classes (the regex crate's \p{L}, \p{N}, \s) ----

def _is_letter(c:str) -> bool: return unicodedata.category(c).startswith("L")
def _is_number(c:str) -> bool: return unicodedata.category(c).startswith("N")

# Rust regex's Unicode \s (Python's isspace() includes a few more control chars)
_SPACE = frozenset("\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000")
def _is_space(c:str) -> bool: return c in _SPACE

_CONTRACTION = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)")

def _pretokenize(text:str) -> list[str]:
  """Qwen's GPT-2 style Split regex, leftmost-first like the Rust regex crate.

  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
  """
  out, i, n = [], 0, len(text)
  while i < n:
    # 1. English contractions
    if (m := _CONTRACTION.match(text, i)) is not None:
      out.append(m.group(0)); i = m.end(); continue
    # 2. an optional non-letter/number char, then a run of letters
    if _is_letter(text[i]):
      j = i+1
      while j < n and _is_letter(text[j]): j += 1
      out.append(text[i:j]); i = j; continue
    if text[i] not in "\r\n" and not _is_number(text[i]) and i+1 < n and _is_letter(text[i+1]):
      j = i+2
      while j < n and _is_letter(text[j]): j += 1
      out.append(text[i:j]); i = j; continue
    # 3. a single number char
    if _is_number(text[i]):
      out.append(text[i]); i += 1; continue
    # 4. optional space, a run of punctuation/symbols, then trailing newlines
    start = i+1 if text[i] == " " else i
    if start < n and not _is_space(text[start]) and not _is_letter(text[start]) and not _is_number(text[start]):
      j = start+1
      while j < n and not _is_space(text[j]) and not _is_letter(text[j]) and not _is_number(text[j]): j += 1
      while j < n and text[j] in "\r\n": j += 1
      out.append(text[i:j]); i = j; continue
    # 5/6/7. whitespace runs: \s*[\r\n]+ then \s+(?!\S) then \s+
    if _is_space(text[i]):
      j = i
      while j < n and _is_space(text[j]): j += 1
      last_nl = max((k for k in range(i, j) if text[k] in "\r\n"), default=-1)
      if last_nl >= 0:
        out.append(text[i:last_nl+1]); i = last_nl+1; continue
      if j == n or j-i >= 2:
        out.append(text[i:j if j == n else j-1]); i = j if j == n else j-1; continue
      out.append(text[i:j]); i = j; continue
    # a lone unmatched char (non-space whitespace classes we missed): keep it as its own piece
    out.append(text[i]); i += 1
  return out

def bytes_to_unicode() -> dict[int, str]:
  """GPT-2's reversible byte -> printable unicode table (what ByteLevel stores in the vocab)."""
  bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
  cs = bs[:]
  n = 0
  for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n += 1
  return dict(zip(bs, map(chr, cs)))

class Tokenizer:
  def __init__(self, tokenizer_json:str|Path|dict):
    d = json.loads(Path(tokenizer_json).read_text(encoding="utf-8")) if isinstance(tokenizer_json, (str, Path)) else tokenizer_json
    model = d["model"]
    assert model["type"] == "BPE", f"only BPE tokenizers are supported, got {model['type']}"
    self.vocab:dict[str, int] = model["vocab"]
    self.ranks:dict[tuple[str, str], int] = {}
    for i, m in enumerate(model["merges"]):
      pair = tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
      assert len(pair) == 2, f"bad merge entry {m!r}"
      self.ranks[pair] = i
    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
    self.added:dict[str, int] = {a["content"]: a["id"] for a in d.get("added_tokens", [])}
    self._added_re = re.compile("|".join(re.escape(t) for t in sorted(self.added, key=len, reverse=True))) if self.added else None
    self._merge_cache:dict[str, tuple[str, ...]] = {}
    self._id_cache:dict[str, list[int]] = {}

  def convert_tokens_to_ids(self, token:str) -> int:
    if (i := self.added.get(token)) is not None: return i
    return self.vocab.get(token, self.vocab.get("<|endoftext|>", 0))

  def _split_added(self, text:str):
    if self._added_re is None: yield text, False; return
    last = 0
    for m in self._added_re.finditer(text):
      if m.start() > last: yield text[last:m.start()], False
      yield m.group(0), True
      last = m.end()
    if last < len(text): yield text[last:], False

  def _bpe(self, piece:str) -> tuple[str, ...]:
    if (hit := self._merge_cache.get(piece)) is not None: return hit
    word = tuple(piece)
    if len(word) > 1:
      pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}
      while len(word) > 1:
        bigram = min(pairs, key=lambda p: self.ranks.get(p, 1 << 62))
        if bigram not in self.ranks: break
        a, b = bigram
        merged, i = [], 0
        while i < len(word):
          if i < len(word)-1 and word[i] == a and word[i+1] == b: merged.append(a+b); i += 2
          else: merged.append(word[i]); i += 1
        word = tuple(merged)
        if len(word) == 1: break
        pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}
    self._merge_cache[piece] = word
    return word

  def _piece_ids(self, piece:str) -> list[int]:
    if (hit := self._id_cache.get(piece)) is not None: return hit
    mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
    ids = []
    for sym in self._bpe(mapped):
      if (i := self.vocab.get(sym)) is None: raise ValueError(f"BPE symbol {sym!r} is not in the vocabulary")
      ids.append(i)
    self._id_cache[piece] = ids
    return ids

  def encode(self, text:str, add_special_tokens:bool=False) -> list[int]:
    ids:list[int] = []
    for chunk, is_added in self._split_added(text):
      if is_added: ids.append(self.added[chunk]); continue
      for piece in _pretokenize(unicodedata.normalize("NFC", chunk)): ids.extend(self._piece_ids(piece))
    return ids

  def decode(self, ids:list[int]) -> str:
    rev = {v: k for k, v in self.added.items()}
    raw = bytearray()
    for i in ids:
      if (t := rev.get(i)) is not None: raw += t.encode("utf-8"); continue
      sym = next((s for s, j in self.vocab.items() if j == i), None)
      if sym is None: continue
      raw += bytes(self.byte_decoder[c] for c in sym)
    return raw.decode("utf-8", errors="replace")
