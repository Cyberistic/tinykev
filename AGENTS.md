# Notes

- tinygrad is the pinned submodule in `vendor/tinygrad`; run everything with `PYTHONPATH=vendor/tinygrad:.`.
- Tests: `python -m unittest discover -s tinykev/test -v`.
- Tokenizer reference test: `TINYKEV_TOKENIZER=/path/to/tokenizer.json python -m unittest tinykev.test.test_tokenizer -v` (needs `tokenizers`).
- Numerical parity: `python scripts/parity_hf.py --run <local run dir> --base <local base dir>` (needs torch, transformers>=5.17, safetensors, numpy).
- `.uir` files under `tinykev/uir/` are generated at runtime and must not be committed.
- Do not amend commits.
