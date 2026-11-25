# tinykev

Runs [Kev](https://github.com/jaredpalmer/kev) (open-source jev) decision models on [tinygrad](https://github.com/tinygrad/tinygrad) at 3x the speed on a macbook (metal).

It's almost faster than my keystrokes and costs very little memory, like ~20ms per decision. 
so fast you can make it type-check everything you write or auto sort your downloads folder
how is it so fast??? im so excited
what? 
vroom vroom tinykevvvvvrooooom kachawww


Kev-0.8B (101 tokens, M2 Pro, fp16, 20-iter medians):

|Backend|ms|
|---|---|
|**tinykev (Metal, cached prefix)**|**19.5** (min 19.1)|
|**tinykev (Metal, cold packed)**|**24.5** (min 24.3)|
|Kev MLX, cached prefix|45.5|
|Kev MLX, row form|58.0|
|Kev MLX, serving `probs`|67.1|

```bash
PYTHONPATH=vendor/tinygrad:. python scripts/bench.py --run jaredpalmer/kev-0.8b --iters 20 --cached-prefix
```

license: MIT 
