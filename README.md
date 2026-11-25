# tinykev

Runs [Kev](https://github.com/jaredpalmer/kev) decision models on [tinygrad](https://github.com/tinygrad/tinygrad) without torch, using packed-sequence Metal UIR kernels (WMMA/Split-K GEMMs, SwiGLU, chunked WY DeltaNet, Flash Attention, fused RMSNorm, readout pruning, cached prefix) via `TinyJit`.

Kev-0.8B (101 tokens, M2 Pro, fp16, 20-iter medians):

|Backend|ms|
|---|---|
|**tinykev (Metal, cached prefix)**|**20.0** (min 19.6)|
|**tinykev (Metal, cold packed)**|**24.6** (min 24.4)|
|Kev MLX, cached prefix|45.5|
|Kev MLX, row form|58.0|
|Kev MLX, serving `probs`|67.1|

```bash
PYTHONPATH=vendor/tinygrad:. python scripts/bench.py --run jaredpalmer/kev-0.8b --iters 20 --cached-prefix
```
