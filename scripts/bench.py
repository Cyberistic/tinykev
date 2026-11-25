"""Benchmark tinykev on a request: warm up the JIT, then report per-request latency.

    PYTHONPATH=vendor/tinygrad:. python scripts/bench.py --run jaredpalmer/kev-0.8b --iters 10

By default the port runs the packed form (every token once; rows only inside the sequence mixers, see
`tinykev/packed.py`); `TINYKEV_PACKED=0` selects the row form (each question is `state + branch`). Compare with `kev.probs`
(prefix path) on the reference backend, and with `kev.forward_rows` for the same algorithm.
"""
from __future__ import annotations

import argparse, json, statistics, time

from tinykev import encode, load_kev, to_answers, to_record
from tinykev.encode import rows_of

REQUEST = {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
           "questions": {
             "department": {"type": "choice", "instructions": "Which team should handle this?",
                            "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                         "shipping": "Delivery status, delays, lost packages",
                                         "billing": "Charges, invoices, payment problems"}},
             "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
             "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                             "criteria": ["Calm", "Frustrated", "Very angry"]}}}

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True)
  p.add_argument("--state", default=REQUEST["state"])
  p.add_argument("--questions", default=json.dumps(REQUEST["questions"]))
  p.add_argument("--max-context", type=int, default=1024)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--warmup", type=int, default=2)
  p.add_argument("--dtype", choices=["f16", "f32"], default="f16")
  p.add_argument("--no-jit", dest="jit", action="store_false", help="rebuild the graph every request (default: JIT replay)")
  p.add_argument("--cold", action="store_true", help="drop the cached state prefix before every request")
  p.set_defaults(jit=True)
  args = p.parse_args()

  from tinygrad import dtypes
  t0 = time.perf_counter()
  tok, kev, meta = load_kev(args.run, max_context=args.max_context, jit=args.jit,
                            dtype=dtypes.float16 if args.dtype == "f16" else dtypes.float32)
  load_s = time.perf_counter() - t0

  rec, qmeta = to_record({"state": args.state, "questions": json.loads(args.questions)})
  enc = encode(tok, rec, max_state=args.max_context, max_branch=args.max_context, strict=True)
  state, _, rows = rows_of(enc)
  print(f"load {load_s:.1f}s  dtype {args.dtype}  state {len(state)} tokens, {len(rows)} rows, "
        f"row tokens {[len(state)+len(r['ids']) for r in rows]}, total {len(enc['ids'])}")

  for _ in range(args.warmup): kev.probs(enc)
  times = []
  for _ in range(args.iters):
    if args.cold: kev.clear_prefix()
    t = time.perf_counter(); probs = kev.probs(enc); times.append(time.perf_counter() - t)
  times.sort()
  print(f"per request: min {times[0]*1000:.1f} ms  median {statistics.median(times)*1000:.1f} ms  "
        f"p90 {times[min(len(times)-1, int(0.9*len(times)))]*1000:.1f} ms  ({args.iters} iters)")
  print(json.dumps(to_answers(probs, qmeta), indent=2))

if __name__ == "__main__": main()
