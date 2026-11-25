# tinykev

Runs [Kev](https://github.com/jaredpalmer/kev) decision models on [tinygrad](https://github.com/tinygrad/tinygrad). There's no torch at runtime. The scan, short conv, GEMMs, gathers and readout are hand-written Metal UIR kernels, and every request is replayed through `TinyJit`. Each token is computed once, in packed order. Only the conv, scan and attention see per-question rows.

Kev-0.8B, 101-token request, Apple Silicon, fp16, medians:

| backend | ms |
|---|---|
| **tinykev (Metal)** | **TBD** |
| Kev MLX, cached prefix | 45.5 |
| Kev MLX, row form | 58.0 |
| Kev MLX, serving `probs` | 67.1 |

```bash
git clone --recurse-submodules <repo> && cd tinykev && export PYTHONPATH=vendor/tinygrad:.
python scripts/bench.py --run jaredpalmer/kev-0.8b --max-context 1024
python -m unittest discover -s tinykev/test
```
