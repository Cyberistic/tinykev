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
* the readout, the Gated DeltaNet scan, the short conv and the projections are UIR kernels in the `slopkernels` style;
  the scan's cross-lane sum uses Metal `simd_shuffle_xor` (no barriers) and the GEMMs use a hand-scheduled WMMA tile;
  one 101-token request went from **5.9 s to 40 ms** (~150x), faster than every MLX path of the reference.

The 4B/9B adapters and the 27B full-weight checkpoint use the same code paths but have not been run here.

## Performance

Kev-0.8B, one request: 22-token state, 3 questions, 101 tokens total, steady state after warmup, on an M-series Mac
(Metal for tinykev, MLX for Kev, CPU for the torch reference; 10-20 iteration medians):

| backend | per request |
|---|---|
| MLX, cached state prefix (`probs_with_prefix`) | **45.5 ms** |
| MLX, row form (`forward_rows`, same algorithm as tinykev) | **58.0 ms** |
| MLX, serving path (`probs`: state once + branch rows) | **67.1 ms** |
| **tinykev, Metal fp16, packed (default)** | **39.8 ms** (cold, `--cold`: 40.2 ms) |
| tinykev, Metal fp16, state cache warm (`TINYKEV_PREFIX=1`, `TINYKEV_PACKED=0`) | 64.6 ms |
| tinykev, Metal fp16, padded rows (`TINYKEV_PACKED=0`) | 69.3 ms |
| torch fp32 CPU (the exact path Kev's published numbers use) | **13.5 s** |

```bash
PYTHONPATH=vendor/tinygrad:. python scripts/bench.py --run jaredpalmer/kev-0.8b --max-context 1024 --iters 10
TINYKEV_PREFIX=1 PYTHONPATH=vendor/tinygrad:. python scripts/bench.py --run jaredpalmer/kev-0.8b --max-context 1024
```

What moved the needle:

| change | effect |
|---|---|
| portable UIR gated-delta scan + TinyJit replay | 5.9 s -> 1.1 s |
| `TC_OPT=2` (tinygrad's Metal tensor cores are off by default outside BEAM) | 1.1 s -> 114 ms |
| one padded batch for all question rows | 114 -> 93 ms |
| hand-scheduled WMMA GEMM for every Linear (`tinykev/gemm.py`) | 93 -> 78 ms |
| UIR short conv with real threadgroups (`tinykev/conv.py`) | 78 -> 73 ms |
| `simd_shuffle_xor` scan reduce, 8 value rows/program | 73 -> 71 ms cold, -> 65 ms warm |
| packed form: token-wise work on the 101 packed tokens, rows only inside the conv/scan/attention (`tinykev/packed.py`) | 69 -> 40 ms |

The packed form beats every MLX path on this request: 40 % under its serving path, 31 % under its row form and 12 %
under its cached prefix. Before it, the padded row form was 6 % behind MLX's serving path; the remaining gap there is tensor-core tiling for the projection shapes at M=126-192: the custom kernels measure 1.5-2x over
tinygrad's schedule in isolation, but the model is partly memory/latency bound, so the end-to-end gain is ~10 %.
`BEAM=1` with a warm disk cache is slower (85 ms); `BEAM=2` exceeds Metal's variant limit; fusing the MLP or the
DeltaNet projections makes the scheduler emit one pathological kernel and is 3x slower; bf16, transposed weights and
the `TC_SELECT`/`TC_MIN_GLOBALS` knobs are neutral or worse.

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

`tinykev/gemm.py` is a hand-scheduled WMMA GEMM used for every `tinygrad.llm.model.Linear`: TC + 4x M upcast + a
per-shape (N upcast, N local) tile, with f32 accumulation in the tensor cores and a cast to f16 on store. The ranges
must be WEAK (LOOP axes are refused by the TC opt), the M upcast is chosen from `ceil(M/8)` (M=192 -> 4, M=144 -> 3),
and `tinykev/uir/gemm_<M>_<K>_<N>.uir` is generated per shape. `tinykev/conv.py` is the DeltaNet short conv as one
elementwise kernel (128-thread workgroups x 8 channel upcast; 2.8x the reference kernel).

`tinykev/gated_delta.py` is the bigger one: it is `slopkernels`' `gated_delta_prefill` layout with the AMD
`ds_swizzle` warp reduce replaced by Metal `simd_shuffle_xor` (`_warp_sum`: five XOR shuffles and adds on the reduced
value, no shared memory and no barrier), reading `TINYKEV_ROW_TILE` value rows per program (default 8). 32 lanes
split the key dimension and each lane holds 4 state values of each row in registers; the token loop is a serial LOOP
range and the two dot products per token are a `REDUCE` over the UPCAST key lane, then the shuffle chain. This took
the scan from ~62 ms of the request to ~7 ms. The hook is deliberately narrow:
`PortableGatedDeltaNetBlock._attention` lets tinygrad's existing `amd_gated_delta` branch run on Metal for the
duration of the call and routes only that kernel name to the portable one, so the conv, gates, norm and layouts are
still tinygrad's. `TINYKEV_DELTA_UIR=0` restores the unrolled fallback.

### Packed form

The row form recomputes the state once per question and pads every row to the longest (rows of 64/37/44 tokens
become 192 GEMM rows for 101 real tokens). But only the sequence mixers need rows; embedding, norms, every projection
and the MLPs are token-wise. `tinykev/packed.py` runs those on Kev's packed sequence (state once, then each branch,
padded to a multiple of 16 so the WMMA M tile divides) and moves activations with a vectorized UIR row gather
(`tinykev/gather.py`) only around the mixers: the DeltaNet qkv projection is gathered into `state + branch` rows for
the short conv and the scan and the scan output is gathered back before the gated norm; attention gathers q/k/v into
rows (positions 0..L-1, causal) and its output back before the gate and `o_proj`. Each token's math is exactly the
row form's, so it matches it to 8e-5 in fp32 (f16: 2e-3, the usual f16 noise). The backbone, the pointer head and the
readout are one `TinyJit` per (rows, row length, packed length, options) shape, and the softmax is done on host.
`TINYKEV_PACKED=0` restores the row forms.

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

* **GEMM tiling is still the wall.** The hand-scheduled GEMM is 1.5-2x per kernel but the model is partly memory and
  latency bound, so the end-to-end gain is ~10 %. A fused DeltaNet layer (projection + conv + scan + norm in one
  kernel) is the next step; the projection fusion done at the Tensor level is slower because the scheduler merges the
  consumer into a pathological 30-range kernel.
* **The scan's VT is `head_v_dim/ROW_TILE`**, not `v_tiles/ROW_TILE`; `kernel_params` is unit-tested because getting
  this wrong silently computes a quarter of the value rows.
* **State-prefix reuse is opt-in** (`TINYKEV_PREFIX=1`). It keeps separate stable cache buffers for the state pass
  (B=1) and the branch pass (B=3), both `TinyJit`-captured, and copies the state into the branch buffers per request.
  The padded branch batch (M=126) is less GEMM-efficient than the full 192-position batch, so it only wins warm.
* Long states still compile one conv/norm graph per shape; the token loop is gone from the recurrence, so compile time
  no longer scales with state length the way it did before the scan kernel.
* `option_isolation` is not implemented.
* A new request shape pays two JIT capture calls; row length is bucketed to 8 and packed length to 16 to limit that.
* Full-weight checkpoints (`meta.weights == "full"`, the 27B) share the loader but were not exercised.
* No HTTP server/streaming; the CLI mirrors the `/v1/systemone` response shape.

## Layout

```
tinykev/tokenizer.py   tokenizer.json -> ids (NFC, Split regex, ByteLevel, BPE, added tokens)
tinykev/encode.py      kev.model.encode / rows_of + /v1/systemone mapping
tinykev/weights.py     downloads, config mapping, HF safetensors + LoRA merge, torch-free head.pt
tinykev/model.py       hidden states + pointer head
tinykev/uir.py         templated UIR readout kernel
tinykev/packed.py      packed inference (token-wise work once per token, rows only in the mixers)
tinykev/gather.py      UIR row gather between packed and row order
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
