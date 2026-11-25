"""Kev's encoder: packing, row splitting and the /v1/systemone mapping, without any model."""
from __future__ import annotations

import unittest

from tinykev.encode import (OPT_DECIDE, OPT_NONE, SPECIAL, ContextOverflow, encode, rows_of, to_answers, to_record)

class FakeTok:
  """Deterministic stand-in for the Qwen tokenizer: one id per character."""
  def __init__(self): self.table, self.next = {}, 100
  def _id(self, s):
    if s not in self.table: self.table[s] = self.next; self.next += 1
    return self.table[s]
  def convert_tokens_to_ids(self, s): return self._id(s)
  def encode(self, text, add_special_tokens=False): return [self._id(c) for c in text]

class TestEncode(unittest.TestCase):
  def setUp(self):
    self.tok = FakeTok()
    self.rec = {"state": "abc", "questions": [
      {"instr": "i", "options": ["x", "yy"], "label": 0},
      {"instr": "j", "options": ["z"], "label": 0}]}
    self.enc = encode(self.tok, self.rec)
    self.q_id, self.o_id, self.c_id, self.d_id = (self.tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    self.s_id = self.tok.convert_tokens_to_ids(SPECIAL[0])

  def test_layout(self):
    enc = self.enc
    self.assertEqual(enc["seg"].count(0), 4)                       # <state> + 3 state tokens
    self.assertEqual(enc["ids"][0], self.s_id)
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
      self.assertEqual(enc["ids"][d], self.d_id)
      self.assertEqual(enc["seg"][d], k)
      self.assertEqual([enc["ids"][i] for i in oi], [self.c_id]*len(oi))
      self.assertEqual([enc["opt"][i] for i in oi], list(range(len(oi))))
      self.assertEqual(enc["opt"][d], OPT_DECIDE)
    self.assertEqual(enc["opt"][0], OPT_NONE)

  def test_rows(self):
    state, pos, rows = rows_of(self.enc)
    self.assertEqual(state, self.enc["ids"][:4])
    self.assertEqual(pos, [0, 1, 2, 3])
    self.assertEqual(len(rows), 2)
    for r, (d, oi) in zip(rows, zip(self.enc["decide_idx"], self.enc["opt_idx"])):
      self.assertEqual(r["ids"][r["decide"]], self.d_id)
      self.assertEqual([r["ids"][o] for o in r["opts"]], [self.c_id]*len(r["opts"]))
      self.assertEqual(r["pos"], list(range(4, 4 + len(r["ids"]))))
    # concatenating the rows back reproduces the branch tokens
    self.assertEqual(state + [t for r in rows for t in r["ids"]], self.enc["ids"])

  def test_strict_overflow(self):
    with self.assertRaises(ContextOverflow): encode(self.tok, {"state": "abcdef", "questions": []}, max_state=4, strict=True)
    self.assertTrue(encode(self.tok, {"state": "abcdef", "questions": []}, max_state=4)["state_truncated"])

  def test_request_mapping(self):
    req = {"state": "hello", "questions": {
      "d": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}},
      "n": {"type": "noul", "instructions": "ok?"},
      "s": {"type": "score", "instructions": "how", "criteria": ["low", "high"]}}}
    rec, meta = to_record(req)
    self.assertEqual(rec["questions"][0]["options"], ["a: A", "b: B"])
    self.assertEqual(rec["questions"][1]["options"], ["no", "yes"])
    self.assertEqual([m["keys"] for m in meta], [["a", "b"], ["false", "true"], ["0", "1"]])
    ans = to_answers([[0.25, 0.75], [0.2, 0.8], [0.9, 0.1]], meta)
    self.assertEqual(ans["d"]["choice"], "b")
    self.assertAlmostEqual(ans["n"]["noul"], 0.8)
    self.assertAlmostEqual(ans["s"]["score"], 0.1)
    self.assertAlmostEqual(ans["s"]["confidence"], 0.8)

if __name__ == "__main__": unittest.main()
