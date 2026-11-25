"""Packed inference (every token once) must match the row form (one causal row per question)."""
from __future__ import annotations

import unittest
from pathlib import Path

from tinygrad import Tensor

from tinykev.model import Kev
from tinykev.packed import layout
from tinykev.weights import Loaded
from tinykev.test.test_model import tiny_model

def _enc(state:list[int], branches:list[tuple[list[int], list[int]]]) -> dict:
  """A packed encoding: branches are (ids, option-end offsets); the last branch token is <decide>."""
  ids, seg, decide, opt_idx = list(state), [0]*len(state), [], []
  for k, (b, ends) in enumerate(branches, start=1):
    base = len(ids)
    ids += b; seg += [k]*len(b)
    decide.append(base + len(b) - 1); opt_idx.append([base + e for e in ends])
  return {"ids": ids, "seg": seg, "pos": [], "decide_idx": decide, "opt_idx": opt_idx}

class TestPacked(unittest.TestCase):
  def setUp(self):
    dev = Tensor.empty(0).device
    head = {"q.weight": Tensor.randn(8, 32, device=dev)*0.1, "q.bias": Tensor.zeros(8, device=dev),
            "k.weight": Tensor.randn(8, 32, device=dev)*0.1, "k.bias": Tensor.zeros(8, device=dev)}
    self.kev = Kev(Loaded(tiny_model(), {"head": head, "temperature": 1.0, "head_dim": 8}, {}, Path(".")), device=dev)
    self.enc = _enc([1, 5, 9, 2], [([7, 3, 11, 4, 6, 8], [2, 4]), ([12, 13, 14], [0, 1]), ([20, 21, 22, 23, 24], [1, 2, 3])])

  def test_layout(self):
    lay = layout(self.enc)
    self.assertEqual((lay["R"], lay["L"]), (3, 16))
    for r in range(lay["R"]):   # every packed token is computed in a row that contains it
      for p, slot in enumerate(lay["pack_idx"][:len(self.enc["ids"])]):
        self.assertEqual(lay["rows_idx"][slot], p)

  def test_packed_matches_rows(self):
    self.kev.use_packed = False
    ref = self.kev.probs(self.enc)
    self.kev.use_packed = True
    got = self.kev.probs(self.enc)
    for a, b in zip(ref, got):
      self.assertEqual(len(a), len(b))
      self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 1e-4)

if __name__ == "__main__": unittest.main()
