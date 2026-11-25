# tinykev

Runs [Kev](https://github.com/jaredpalmer/kev) decision models on [tinygrad](https://github.com/tinygrad/tinygrad) with no torch at runtime, using packed-sequence Metal UIR kernels (WMMA/Split-K GEMMs, SwiGLU, chunked WY Gated DeltaNet, Flash Attention, fused RMSNorm) replayed via `TinyJit`.

Kev-0.8B (101 tokens, M2 Pro, fp16, 20-iter medians):

|Backend|ms|
|---|---|
|**tinykev (Metal)**|**25.6** (min 24.9)|
|Kev MLX, cached prefix|45.5|
|Kev MLX, row form|58.0|
|Kev MLX, serving `probs`|67.1|

```bash
PYTHONPATH=vendor/tinygrad:. python scripts/bench.py --run jaredpalmer/kev-0.8b --iters 20
```
