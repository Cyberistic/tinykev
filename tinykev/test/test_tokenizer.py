"""The dependency-free tokenizer: pre-tokenizer edge cases, BPE, added tokens, and (optionally) a comparison
against HF `tokenizers` for a real Qwen `tokenizer.json`.

Set `TINYKEV_TOKENIZER=/path/to/tokenizer.json` to run the reference comparison; it is skipped otherwise.
"""
from __future__ import annotations

import os, random, tempfile, unittest
from pathlib import Path

from tinykev.tokenizer import Tokenizer, _pretokenize, bytes_to_unicode

class TestPretokenize(unittest.TestCase):
  def test_cases(self):
    self.assertEqual(_pretokenize("hi there"), ["hi", " there"])
    self.assertEqual(_pretokenize("1a"), ["1", "a"])
    self.assertEqual(_pretokenize("don't"), ["don", "'t"])
    self.assertEqual(_pretokenize("a  b"), ["a", " ", " b"])
    self.assertEqual(_pretokenize("..."), ["..."])
    self.assertEqual(_pretokenize("a\n\nb"), ["a", "\n\n", "b"])

  def test_lossless(self):
    alphabet = "abc XYZ019 \t\n.,'!?-_:;\"()[]{}<>|/\\@#$%^&*+=~`éü日😀"
    random.seed(0)
    for _ in range(500):
      s = "".join(random.choice(alphabet) for _ in range(random.randint(0, 40)))
      self.assertEqual("".join(_pretokenize(s)), s)

class TestSyntheticBPE(unittest.TestCase):
  def setUp(self):
    # every single byte is a token, plus the merge h+e -> he; no normalizer or added tokens except one special
    enc = bytes_to_unicode()
    vocab = {ch: i for i, ch in enumerate(enc.values())}
    vocab["he"] = len(vocab)
    d = {"model": {"type": "BPE", "vocab": vocab, "merges": [["h", "e"]]},
         "added_tokens": [{"content": "<|im_end|>", "id": len(vocab)+1, "special": True}],
         "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
           {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated"},
           {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
         "decoder": {"type": "ByteLevel"}}
    self.dir = tempfile.TemporaryDirectory()
    p = Path(self.dir.name) / "tokenizer.json"
    p.write_text(__import__("json").dumps(d))
    self.tok = Tokenizer(p)
    self.vocab = vocab

  def tearDown(self): self.dir.cleanup()

  def test_bpe_and_added(self):
    ids = self.tok.encode("he")
    self.assertEqual(ids, [self.vocab["he"]])
    self.assertEqual(self.tok.encode("hi"), [self.vocab["h"], self.vocab["i"]])
    self.assertEqual(self.tok.encode("a<|im_end|>b"), [self.vocab["a"], len(self.vocab)+1, self.vocab["b"]])
    self.assertEqual(self.tok.convert_tokens_to_ids("<|im_end|>"), len(self.vocab)+1)
    self.assertEqual(self.tok.decode(ids), "he")

@unittest.skipUnless(os.environ.get("TINYKEV_TOKENIZER"), "set TINYKEV_TOKENIZER to a tokenizer.json")
class TestQwenReference(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    try: from tokenizers import Tokenizer as RefTok
    except ImportError: raise unittest.SkipTest("pip install tokenizers")
    cls.path = os.environ["TINYKEV_TOKENIZER"]
    cls.tok, cls.ref = Tokenizer(cls.path), RefTok.from_file(cls.path)

  def test_corpus(self):
    cases = ["Hello world", "The customer's order #A-1934 was 3 days late.",
             "  leading and trailing   ", "don't won't it's I'M we'LL", "héllo wörld 日本語 😀",
             "a\n\nb\r\nc\t d", "<|im_start|>hi<|im_end|>", '{"a":1,"b":[true,null]}']
    alphabet = "abc XYZ019 \t\n.,'!?-_:;\"()[]{}<>|/\\@#$%^&*+=~`éü日😀"
    random.seed(0)
    cases += ["".join(random.choice(alphabet) for _ in range(random.randint(0, 60))) for _ in range(1000)]
    for text in cases:
      self.assertEqual(self.tok.encode(text), self.ref.encode(text, add_special_tokens=False).ids, repr(text))
    for text in cases[:8]: self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

if __name__ == "__main__": unittest.main()
