"""`python -m tinykev --run jaredpalmer/kev-0.8b --state "..." --questions '{...}'`."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

from . import encode, load_kev, to_answers, to_record

def main(argv:list[str]|None=None) -> int:
  p = argparse.ArgumentParser(prog="tinykev", description="Run a Kev decision model on tinygrad")
  p.add_argument("--run", required=True, help="Hub id (jaredpalmer/kev-0.8b), id@revision, or a local run directory")
  p.add_argument("--state", help="document text (or @path to a file)")
  p.add_argument("--questions", help="questions object of a /v1/systemone request (or @path to a .json file)")
  p.add_argument("--request", help="@path to a full /v1/systemone request JSON")
  p.add_argument("--max-context", type=int, default=8192, help="KV cache size; at least the longest state+branch row")
  p.add_argument("--device", default=None)
  p.add_argument("--jit", action="store_true", help="replay row shapes through TinyJit (better for many requests)")
  args = p.parse_args(argv)

  if args.request:
    req = json.loads(Path(args.request[1:] if args.request.startswith("@") else args.request).read_text())
  else:
    if not args.state or not args.questions: p.error("give --request, or both --state and --questions")
    state = args.state
    if state.startswith("@"): state = Path(state[1:]).read_text()
    questions = args.questions
    if questions.startswith("@"): questions = Path(questions[1:]).read_text()
    req = {"state": state, "questions": json.loads(questions)}

  rec, meta = to_record(req)
  tok, kev, ck = load_kev(args.run, max_context=args.max_context, device=args.device, jit=args.jit)
  enc = encode(tok, rec, max_state=args.max_context, max_branch=args.max_context, strict=True)
  probs = kev.probs(enc)
  print(json.dumps({"model": req.get("model", "kev"), "answers": to_answers(probs, meta),
                    "usage": {"state_tokens": enc["state_tokens"], "total_tokens": len(enc["ids"])}}, indent=2))
  return 0

if __name__ == "__main__": sys.exit(main())
