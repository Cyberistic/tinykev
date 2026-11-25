# tinykev

Run [Kev](https://github.com/jaredpalmer/kev) decision models on [tinygrad](https://github.com/tinygrad/tinygrad).

Kev is a family of Qwen3.5 / Qwen3.8 based decision models. They never generate text: they read a document and a
set of questions (`yes/no`, `choice`, `score`) and a small pointer head scores each option from the backbone's hidden
states. This repo ports the reference torch/MLX inference path to tinygrad — tokenizer, packing, weights, LoRA merge,
head and serving CLI — with no torch or transformers at runtime.

```json
{
  "department": { "choice": "shipping", "probabilities": { "returns": 0.2609, "shipping": 0.4592, "billing": 0.2799 } },
  "escalate":   { "noul": 0.5541 },
  "frustration":{ "score": 1.228 }
}
```

## Status

Working end to end on Metal with `jaredpalmer/kev-0.8b`:

* the byte-level BPE tokenizer read straight from `tokenizer.json` (no `tokenizers` dependency), **0 mismatches**
  against HF `tokenizers` on 3,000 fuzz strings plus unicode/contraction/whitespace cases;
* HF safetensors loading with PEFT LoRA merged in, and `head.pt` read without torch (bit-exact against `torch.load`);
* fp32 hidden states vs HF `transformers` 5.24 on the same inputs: **max |diff| 0.06, mean 0.0055** (reference
  activations have mean |x| ≈ 1.69), and question probabilities agree to **≈1e-3**;
* fp16 weights reproduce fp32 probabilities to **<2e-4**;
* the readout is a templated **UIR kernel**, generated the same way `slopkernels` generates its AMD kernels.

The 4B/9B adapters and the 27B full-weight checkpoint use the same code paths but have not been run here.

## Quick start

```bash
git clone --recurse-submodules <this repo> tinykev && cd tinykev
export PYTHONPATH=vendor/tinygrad:.

python -m tinykev --run jaredpalmer/kev-0.8b --max-context 1024 \
  --state "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card." \
  --questions '{
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing": "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}}'
```

`--run` also accepts `repo@revision` or a local run directory. Weights, adapter, tokenizer and head are cached in
`~/.cache/tinykev` (`TINYKEV_CACHE` overrides). `--max-context` sizes the KV cache and must be at least the longest
`state + branch` row; it is not the backbone's 262k rotary limit.

Python API:

```python
from tinykev import encode, load_kev, to_answers, to_record

tok, kev, meta = load_kev("jaredpalmer/kev-0.8b", max_context=8192)
rec, qmeta = to_record(request)                       # a /v1/systemone request body
enc = encode(tok, rec, max_state=8192, max_branch=8192)
print(to_answers(kev.probs(enc), qmeta))
```

## How it works

### One causal row per question

Kev packs `[<state>] state ... [<q>] instruction [<opt>] option0 [/opt] ... [<decide>]` per question and blocks
cross-question attention. Qwen3.5/3.8 have Gated DeltaNet layers whose recurrent state cannot honour an arbitrary
mask, so Kev always uses the equivalent **row form**: one causal row per question, `state + branch`, all rows
independent. That is what this port implements:

```
state + branch_k  ->  Transformer blocks  ->  output_norm  ->  h[<decide>], h[</opt>_j]
                                                                    |
                                                     pointer head:  dot(q(h_dec), k(h_opt)) / sqrt(dp) / T
                                                                    |
                                                            softmax over options
```

`encode.py` is a line-for-line port of `kev.model.encode` / `rows_of` (including the serving context limits and the
`<¦name¦>` rewrite that makes option boundaries unforgeable). `model.py` walks `tinygrad.llm.model.Transformer`'s
blocks itself instead of calling `forward()` (which samples a token) so it can keep `output_norm(x)` for every row.

### Weights

`tinygrad.llm.model.Transformer` already implements Qwen3.5's hybrid stack from GGUF. Loading HF safetensors directly
needs three Transformers-specific adjustments, all in `weights.py`:

1. `Qwen3_5RMSNorm` stores `1 + w`; the DeltaNet's gated RMSNorm (`linear_attn.norm`) does not. The normalized
   weights get `+1` only for the former.
2. Kev's recurrence reads `ssm_a = -exp(A_log)`; `A_log` is transformed on load.
3. `q_proj` is `[heads, 2*head_dim]` per head (query then output gate), which is exactly tinygrad's
   `attn_output_gate` layout, and the DeltaNet `conv1d.weight` is `[C, 1, K]` and needs its middle dim squeezed.

PEFT adapters are folded in fp32 (`W + (B @ A) * alpha / r`) and rounded once to the loaded dtype. The LM head is
never used, so `output.weight` is tied to the embeddings instead of copying 500 MB.

### Tokenizer and head

`tokenizer.py` implements the pieces of HF `tokenizers` that Qwen's `tokenizer.json` uses: NFC normalization, the
GPT-2-style `Split` regex (including unicode letter/number classes via `unicodedata`), `ByteLevel` mapping, the BPE
merge table, and added/special tokens. `head.pt` is a `torch.save` zip; `weights.read_meta` reads it with `pickle`,
`zipfile` and `array` — no torch.

### UIR kernel

`tinykev/uir.py` writes Kev's readout as a pure UOp graph, renders it to `tinykev/uir/pointer_readout.uir`, parses it
back and binds shapes with `specialize()`, exactly the flow `vendor/tinygrad/slopkernels` uses for its AMD kernels:

```
%0 = param [] slot=-1 dtype=weakint vmin_vmax=[1 1048576] name=M addrspace=TEMPLATE
...
%23 = reduce %22 %13 op=add
%24 = store %3 %23
```

The kernel fuses the gather of the option rows against the projected query and the reduction
(`sum_p Q[owner[m], p] * K[m, p]`); the two projections stay as matmuls. Unlike the slopkernels kernels this one is
not AMD-specific — it is ordinary tinygrad IR and runs on Metal/CPU too. `TINYKEV_UIR=0` forces the Tensor-op path,
which is the fallback if the kernel cannot be built.

## Validation

`scripts/parity_hf.py` loads the same checkpoint twice — tinykev and HF `transformers` + a manually merged adapter —
and prints both hidden-state and probability differences.

```bash
PYTHONPATH=vendor/tinygrad:. python scripts/parity_hf.py \
  --run ~/.cache/tinykev/jaredpalmer/kev-0.8b/main \
  --base ~/.cache/tinykev/Qwen/Qwen3.5-0.8B-Base/<revision>
```

Observed on an M-series Mac, Kev-0.8B, fp32:

```
hidden max|diff| 0.06171  mean 0.005500  ref scale 1.694
tinykev probs [[0.2611, 0.4589, 0.2801], [0.4456, 0.5544], [0.0514, 0.6692, 0.2794]]
hf probs      [[0.2612, 0.4579, 0.2809], [0.4457, 0.5543], [0.0512, 0.6692, 0.2796]]
```

The residual hidden-state difference comes from tinygrad's unrolled fp32 delta-rule reference versus HF's chunked
reference; probabilities are unaffected at this scale.

## Limitations / next steps

* **No prefix cache reuse.** Each question recomputes the state row (`kev.forward_rows_batch`'s exact form). Kev's
  serving path runs the state once and copies the recurrent/KV cache per question; tinygrad's blocks write caches in
  place, so that needs an explicit snapshot/restore.
* **Metal compiles the DeltaNet scan unrolled.** Rule-of-thumb on this repo's tinygrad: one Python-level step per token
  per DeltaNet layer. Long states are slow to compile. The natural next UIR kernel is a chunked gated-delta prefill
  (the shape of `slopkernels/src/amd.py::_gated_delta_prefill_kernel`), which would make 4k+ token states practical.
* `option_isolation` is not implemented: it needs the packed block-causal mask, which hybrid backbones cannot use.
* Full-weight checkpoints (`meta.weights == "full"`, the 27B) share the loader but were not exercised.
* No HTTP server/streaming; the CLI mirrors the `/v1/systemone` response shape.

## Layout

```
tinykev/tokenizer.py   tokenizer.json -> ids (NFC, Split regex, ByteLevel, BPE, added tokens)
tinykev/encode.py      kev.model.encode / rows_of + /v1/systemone mapping
tinykev/weights.py     downloads, config mapping, HF safetensors + LoRA merge, torch-free head.pt
tinykev/model.py       hidden states + pointer head
tinykev/uir.py         templated UIR readout kernel
tinykev/test/          tokenizer, encoder, tiny-model causality/head, UIR kernel
scripts/parity_hf.py   numerical parity against HF transformers
vendor/tinygrad        pinned tinygrad submodule
```

## Development

tinygrad is pinned at the submodule commit this port was written against (`ab98ebd`, the tip of `origin/master` when
the work started).

```bash
export PYTHONPATH=vendor/tinygrad:.
python -m unittest discover -s tinykev/test -v
TINYKEV_TOKENIZER=/path/to/tokenizer.json python -m unittest tinykev.test.test_tokenizer -v   # needs `tokenizers`
```

## License

Apache-2.0, matching Kev. The encoder, tokenizer and checkpoint layout are ports of Apache-2.0 Kev code.
