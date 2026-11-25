"""Numerical parity check: tinykev vs HF transformers (fp32 CPU) for the same Kev checkpoint.

Needs the reference environment: torch, transformers>=5.17, safetensors.

    PYTHONPATH=vendor/tinygrad:. python scripts/parity_hf.py \
      --run ~/.cache/tinykev/jaredpalmer/kev-0.8b/main \
      --base ~/.cache/tinykev/Qwen/Qwen3.5-0.8B-Base/<revision>

Prints the max |diff| of the final-norm hidden states and of the question probabilities.
"""
from __future__ import annotations

import argparse, json, math

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from tinykev import encode, load_kev, to_record


def hf_load(run:str, base:str, dtype=torch.float32):
  model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, attn_implementation="eager").eval()
  cfg = json.load(open(f"{run}/adapter_config.json"))
  alpha = cfg["lora_alpha"] / ((cfg["r"] ** 0.5) if cfg.get("use_rslora") else cfg["r"])
  params = dict(model.named_parameters())
  sd = load_file(f"{run}/adapter_model.safetensors")
  with torch.no_grad():
    for name, a in sd.items():
      if not name.endswith(".lora_A.weight"): continue
      stem = name[: -len(".lora_A.weight")]
      target = stem.replace("base_model.model.", "model.", 1) + ".weight"
      params[target].add_((sd[stem + ".lora_B.weight"] @ a) * alpha)
  return model


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--run", required=True)
  p.add_argument("--base", required=True)
  p.add_argument("--text", default="The shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.")
  args = p.parse_args()

  from tinygrad import dtypes
  # force fp32 on the tinygrad side too: fp16 weights would hide mapping bugs behind quantization noise
  tok, kev, meta = load_kev(args.run, max_context=1024, dtype=dtypes.float32)
  ids = tok.encode(args.text)
  h_tg = kev.hidden(ids)
  print(f"tokens {len(ids)}  hidden {tuple(h_tg.shape)}")

  hf = hf_load(args.run, args.base)
  with torch.no_grad():
    out = hf.model(input_ids=torch.tensor([ids]), position_ids=torch.arange(len(ids)).unsqueeze(0), use_cache=False)
  h_ref = out.last_hidden_state[0]
  a, b = h_tg.numpy(), h_ref.numpy()
  diff = np.abs(a - b)
  print(f"hidden max|diff| {diff.max():.5f}  mean {diff.mean():.6f}  ref scale {np.abs(b).mean():.3f}")

  rec, qmeta = to_record({"state": args.text, "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"}},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}}})
  enc = encode(tok, rec, max_state=1024, max_branch=1024)
  print("tinykev probs", [[round(x, 4) for x in q] for q in kev.probs(enc)])

  # the same rows through the reference model, then Kev's pointer head in torch
  from tinykev.encode import rows_of
  state_ids, _, rows = rows_of(enc)
  head = torch.load(f"{args.run}/head.pt", map_location="cpu")["head"]
  temp = float(torch.load(f"{args.run}/head.pt", map_location="cpu")["temperature"])
  with torch.no_grad():
    ps = []
    for r in rows:
      o = hf.model(input_ids=torch.tensor([state_ids + r["ids"]]), position_ids=torch.arange(len(state_ids) + len(r["ids"])).unsqueeze(0), use_cache=False)
      h = o.last_hidden_state[0]
      base = len(state_ids)
      q = h[base + r["decide"]] @ head["q.weight"].T + head["q.bias"]
      k = h[torch.tensor([base + o for o in r["opts"]])] @ head["k.weight"].T + head["k.bias"]
      z = (k * q).sum(-1) / math.sqrt(head["q.weight"].shape[0]) / temp
      ps.append(z.softmax(-1).tolist())
  print("hf probs     ", [[round(x, 4) for x in q] for q in ps])

if __name__ == "__main__": main()
