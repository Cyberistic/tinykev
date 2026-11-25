"""Checkpoint loading: HF Qwen3.5/3.8 safetensors, PEFT LoRA merge, and Kev's `head.pt`.

The Kev checkpoints are either a PEFT adapter on a Qwen base (`meta.weights == "lora"`, the 0.8B/4B/9B) or
the whole backbone (`"full"`, the 27B).  Either way the LM head is never used: Kev reads hidden states.
Weights are mapped onto `tinygrad.llm.model.Transformer` (which already implements Qwen3.5's hybrid
Gated DeltaNet / gated-attention stack from GGUF) with three Transformers-specific fixes:

  * `Qwen3_5RMSNorm` stores `1 + w`, so the normalized weight is `w + 1` (KKev's Gated RMSNorm does not).
  * `ssm_a` is `-exp(A_log)` in Kev's recurrence, so the raw `A_log` is transformed on load.
  * `q_proj` is `[H, 2*head_dim]` per head (query then output gate), which is exactly the layout tinygrad's
    `attn_output_gate` expects, and the DeltaNet `conv1d` is `[C, 1, K]` and needs its middle dim squeezed.

`head.pt` is a `torch.save` zip; we read it with the standard library so the port needs neither torch nor
safetensors.
"""
from __future__ import annotations

import array, json, math, os, pathlib, pickle, shutil, struct, urllib.error, urllib.request, zipfile
from dataclasses import dataclass

from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.hf import hf_load
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig

CACHE = pathlib.Path(os.environ.get("TINYKEV_CACHE", pathlib.Path.home() / ".cache" / "tinykev"))
BASE_PREFIX = "model.language_model."
ADAPTER_PREFIX = "base_model.model."
# PEFT wraps the text model; adapter stems are `layers.N.<module>` relative to this
ADAPTER_ROOT = "model.language_model."

# ---- downloads ----

def _fetch(url:str, dest:pathlib.Path) -> pathlib.Path:
  if dest.exists() and dest.stat().st_size > 0: return dest
  dest.parent.mkdir(parents=True, exist_ok=True)
  tmp = dest.with_name(dest.name + ".tmp")
  with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f: shutil.copyfileobj(r, f, 1 << 20)
  os.replace(tmp, dest)
  return dest

def _try_fetch(url:str, dest:pathlib.Path) -> pathlib.Path|None:
  try: return _fetch(url, dest)
  except urllib.error.HTTPError as e:
    if e.code == 404: return None
    raise

def hub_dir(repo:str, revision:str|None, files:list[str]) -> pathlib.Path:
  d = CACHE / repo / (revision or "main")
  for f in files: _fetch(f"https://huggingface.co/{repo}/resolve/{revision or 'main'}/{f}", d / f)
  return d

def _hub_shards(d:pathlib.Path, repo:str, rev:str|None) -> None:
  base = f"https://huggingface.co/{repo}/resolve/{rev or 'main'}"
  idx = _try_fetch(base + "/model.safetensors.index.json", d / "model.safetensors.index.json")
  if idx is not None:
    for s in sorted(set(json.loads(idx.read_text())["weight_map"].values())): _fetch(f"{base}/{s}", d / s)
  else:
    _try_fetch(base + "/model.safetensors", d / "model.safetensors")

def resolve(run:str) -> tuple[pathlib.Path, dict]:
  """Local run directory, or a Hub repo id (`jaredpalmer/kev-0.8b`, optionally `@revision`).  Returns the
  directory and its metadata dict.  Downloads the tokenizer, head and whichever checkpoint layout the repo
  uses (PEFT adapter or full-weight shards)."""
  if os.path.isdir(run):
    d = pathlib.Path(run)
  else:
    repo, _, rev = run.partition("@")
    d = CACHE / repo / (rev or "main")
    base = f"https://huggingface.co/{repo}/resolve/{rev or 'main'}"
    _fetch(base + "/tokenizer.json", d / "tokenizer.json")
    _fetch(base + "/head.pt", d / "head.pt")
    if _try_fetch(base + "/adapter_model.safetensors", d / "adapter_model.safetensors") is None:
      # full-weight checkpoint: its own text config and the backbone shards
      _fetch(base + "/config.json", d / "config.json")
      _hub_shards(d, repo, rev or None)
    else:
      _fetch(base + "/adapter_config.json", d / "adapter_config.json")
  if not (d / "head.pt").exists(): raise FileNotFoundError(f"{d} has no head.pt")
  return d, read_meta(d / "head.pt")

def resolve_base(meta:dict, revision:str|None=None) -> pathlib.Path:
  repo = meta["base"]
  rev = revision or meta.get("base_revision")
  d = CACHE / repo / (rev or "main")
  _fetch(f"https://huggingface.co/{repo}/resolve/{rev or 'main'}/config.json", d / "config.json")
  if not ((d / "model.safetensors.index.json").exists() or (d / "model.safetensors").exists()): _hub_shards(d, repo, rev)
  return d

# ---- head.pt (torch.save zip, no torch) ----

class _Storage:
  __slots__ = ("key", "dtype", "numel")
  def __init__(self, key:str, dtype:str, numel:int): self.key, self.dtype, self.numel = key, dtype, numel

class _Rebuild:
  __slots__ = ("storage", "offset", "size", "stride")
  def __init__(self, storage:_Storage, offset:int, size:tuple, stride:tuple):
    self.storage, self.offset, self.size, self.stride = storage, offset, size, stride

class _Dict(dict):
  def __setstate__(self, state):
    if isinstance(state, tuple): state = state[0]
    if isinstance(state, dict): self.update(state)

class _Unpickler(pickle.Unpickler):
  def find_class(self, module:str, name:str):
    if module == "collections" and name == "OrderedDict": return _Dict
    if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
      return lambda storage, offset, size, stride, *a: _Rebuild(storage, offset, tuple(size), tuple(stride))
    if module == "torch" and name.endswith("Storage"): return name
    raise pickle.UnpicklingError(f"head.pt references {module}.{name}; not supported")
  def persistent_load(self, pid):
    _, storage_type, key, _location, numel = pid
    dtype = storage_type if isinstance(storage_type, str) else getattr(storage_type, "__name__", "FloatStorage")
    return _Storage(str(key), dtype.replace("Storage", "").lower(), int(numel))

def _floats(raw:bytes, dtype:str) -> array.array:
  if dtype == "float": return array.array("f", raw)
  if dtype == "half": return array.array("e", raw)
  if dtype == "double": return array.array("d", raw)
  if dtype == "bfloat16":
    out = array.array("f")
    for (h,) in __import__("struct").iter_unpack("<H", raw): out.append(__import__("struct").unpack("<f", bytes((0, 0)) + h.to_bytes(2, "little"))[0])
    return out
  raise ValueError(f"unsupported head storage dtype {dtype!r}")

def read_meta(path:str|pathlib.Path) -> dict:
  """Load `head.pt`: the top-level dict (base, temperature, head_dim, ...) with `head` as tinygrad Tensors."""
  with zipfile.ZipFile(path) as z:
    prefix = z.namelist()[0].split("/")[0]
    meta = _Unpickler(z.open(f"{prefix}/data.pkl")).load()
    def tensor(r:_Rebuild) -> Tensor:
      flat = _floats(z.read(f"{prefix}/data/{r.storage.key}"), r.storage.dtype)
      numel = math.prod(r.size) if r.size else 1
      contiguous = r.stride == tuple(math.prod(r.size[i+1:]) for i in range(len(r.size))) or r.size == ()
      if not contiguous: raise ValueError(f"non-contiguous head tensor with stride {r.stride}")
      return Tensor(list(flat[r.offset:r.offset+numel]), dtype=dtypes.float32).reshape(r.size)
    if "head" in meta: meta["head"] = {k: tensor(v) for k, v in meta["head"].items() if isinstance(v, _Rebuild)}
  return meta

# ---- config + name mapping ----

def text_config(cfg:dict) -> dict:
  """The text tower's config: full checkpoints are `qwen3_5_text`, Hub bases nest it under `text_config`."""
  return cfg.get("text_config", cfg)

def transformer_config(tc:dict, max_context:int) -> TransformerConfig:
  layers = tc["layer_types"]
  rope = tc.get("rope_parameters", {})
  head_dim = tc["head_dim"]
  value_heads, value_dim = tc["linear_num_value_heads"], tc["linear_value_head_dim"]
  ssm = SSMConfig(conv_kernel=tc["linear_conv_kernel_dim"], state_size=tc["linear_key_head_dim"],
                  group_count=tc["linear_num_key_heads"], time_step_rank=value_heads,
                  inner_size=value_heads * value_dim, silu_gate=tc.get("hidden_act", "silu") == "silu")
  return TransformerConfig(
    num_blocks=tc["num_hidden_layers"], dim=tc["hidden_size"], hidden_dim=tc["intermediate_size"],
    n_heads=tc["num_attention_heads"], n_kv_heads=tc["num_key_value_heads"], norm_eps=tc["rms_norm_eps"],
    vocab_size=tc["vocab_size"], head_dim=head_dim, v_head_dim=head_dim,
    rope_theta=rope.get("rope_theta", 10000.0), rope_dim=int(head_dim * rope.get("partial_rotary_factor", 1.0)),
    max_context=max_context, ssm=ssm, ssm_layers=tuple(t != "full_attention" for t in layers),
    attn_output_gate=tc.get("attn_output_gate", False))

def hf_sources(tc:dict, prefix:str=BASE_PREFIX) -> dict[str, str]:
  """tinygrad name -> HF safetensors name.  `mtp.*` and the vision tower are never read.  Hub bases nest the
  text tower under `model.language_model.`; a full `save_pretrained` of the text model has no prefix."""
  names = {"token_embd.weight": prefix + "embed_tokens.weight", "output_norm.weight": prefix + "norm.weight"}
  for i, lt in enumerate(tc["layer_types"]):
    p, t = f"{prefix}layers.{i}.", f"blk.{i}."
    names[t+"attn_norm.weight"] = p + "input_layernorm.weight"
    names[t+"ffn_norm.weight"] = p + "post_attention_layernorm.weight"
    for tg, hf in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
      names[t+f"ffn_{tg}.weight"] = p + f"mlp.{hf}.weight"
    if lt == "linear_attention":
      names.update({t+"attn_qkv.weight": p+"linear_attn.in_proj_qkv.weight", t+"attn_gate.weight": p+"linear_attn.in_proj_z.weight",
                    t+"ssm_beta.weight": p+"linear_attn.in_proj_b.weight", t+"ssm_alpha.weight": p+"linear_attn.in_proj_a.weight",
                    t+"ssm_conv1d.weight": p+"linear_attn.conv1d.weight", t+"ssm_dt.bias": p+"linear_attn.dt_bias",
                    t+"ssm_a": p+"linear_attn.A_log", t+"ssm_norm.weight": p+"linear_attn.norm.weight",
                    t+"ssm_out.weight": p+"linear_attn.out_proj.weight"})
    else:
      names.update({t+"attn_q.weight": p+"self_attn.q_proj.weight", t+"attn_k.weight": p+"self_attn.k_proj.weight",
                    t+"attn_v.weight": p+"self_attn.v_proj.weight", t+"attn_output.weight": p+"self_attn.o_proj.weight",
                    t+"attn_q_norm.weight": p+"self_attn.q_norm.weight", t+"attn_k_norm.weight": p+"self_attn.k_norm.weight"})
  return names

# RMSNorm weights are stored zero-centred (`1 + w`), except the Gated RMSNorm inside the DeltaNet mixer
_ZERO_CENTERED = ("attn_norm.weight", "ffn_norm.weight", "output_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight")

def base_state_dict(base_dir:pathlib.Path, tc:dict, prefix:str=BASE_PREFIX) -> dict[str, Tensor]:
  sources = hf_sources(tc, prefix)
  sd = hf_load(base_dir, ("CPU",), sources, {})
  for name, t in list(sd.items()):
    if name.endswith("ssm_conv1d.weight"): sd[name] = t.squeeze(1)
    elif name == "ssm_a" or name.endswith(".ssm_a"): sd[name] = -t.float().exp()
    elif name.endswith(_ZERO_CENTERED): sd[name] = t.float() + 1.0
  return sd

def merge_lora(sd:dict[str, Tensor], adapter:pathlib.Path, tc:dict, device:str) -> int:
  """Fold a PEFT adapter into the state dict: `W + (B @ A) * alpha / r`, fp32, rounded once to the base dtype."""
  cfg_path = adapter / "adapter_config.json"
  if not cfg_path.exists(): raise FileNotFoundError(f"{adapter} has an adapter but no adapter_config.json")
  acfg = json.loads(cfg_path.read_text())
  alpha = acfg["lora_alpha"] / ((acfg["r"] ** 0.5) if acfg.get("use_rslora") else acfg["r"])
  hf_to_tg = {v: k for k, v in hf_sources(tc).items()}
  weights = nn.state.safe_load(adapter / "adapter_model.safetensors")
  merged = 0
  for name, a in list(weights.items()):
    if not name.endswith(".lora_A.weight"): continue
    stem = name[: -len(".lora_A.weight")]
    if not stem.startswith(ADAPTER_PREFIX): raise ValueError(f"unexpected adapter tensor {name}")
    hf = ADAPTER_ROOT + stem[len(ADAPTER_PREFIX):] + ".weight"
    if (tg := hf_to_tg.get(hf)) is None: raise ValueError(f"adapter tensor {name} has no target in the model (looked for {hf})")
    b = weights[stem + ".lora_B.weight"]
    delta = (b.to(device).float() @ a.to(device).float()) * alpha
    sd[tg] = (sd[tg].to(device).float() + delta).cast(sd[tg].dtype)
    merged += 1
  return merged

class _FusedFFN:
  """`FFNBlock` with the gate and up projections fused into one GEMM (Kev's fused_qwen35 does the same on CUDA)."""
  def _feed_forward(self, x:Tensor, tokens:Tensor|None=None) -> Tensor:
    from tinygrad.llm.model import gated_activation
    gu = self._gate_up(x)
    hd = self.config.hidden_dim
    return self.ffn_down(gated_activation(gu[..., :hd], gu[..., hd:], self.config))

def fuse_mlp(model) -> int:
  """Concatenate `ffn_gate`/`ffn_up` weights and route the block through a single projection."""
  classes:dict[type, type] = {}
  n = 0
  for blk in model.blk:
    if not (hasattr(blk, "ffn_gate") and hasattr(blk, "ffn_up")) or hasattr(blk, "_gate_up"): continue
    if getattr(blk.config, "num_experts", 0): continue
    base = type(blk)
    if base not in classes:
      classes[base] = type(f"Fused{base.__name__}", (_FusedFFN, base), {})
    from tinygrad.llm.model import Linear as LLMLinear
    blk._gate_up = LLMLinear(blk.ffn_gate.weight.shape[1], blk.ffn_gate.weight.shape[0] + blk.ffn_up.weight.shape[0], bias=False)
    blk._gate_up.weight = Tensor.cat(blk.ffn_gate.weight, blk.ffn_up.weight, dim=0).contiguous().realize()
    blk.__class__ = classes[base]
    n += 1
  return n

def fuse_delta_proj(model) -> int:
  """Concatenate each DeltaNet block's qkv/gate/beta/alpha projections into one N=8224 GEMM."""
  n = 0
  for blk in model.blk:
    if not (hasattr(blk, "attn_qkv") and hasattr(blk, "attn_gate")) or hasattr(blk, "_proj"): continue
    if getattr(blk.config.ssm, "kda", False): continue
    ws = (blk.attn_qkv.weight, blk.attn_gate.weight, blk.ssm_beta.weight, blk.ssm_alpha.weight)
    ends, total = [], 0
    for w in ws[:-1]:
      total += w.shape[0]; ends.append(total)
    from tinygrad.llm.model import Linear as LLMLinear
    lin = LLMLinear(ws[0].shape[1], sum(w.shape[0] for w in ws), bias=False)
    lin.weight = Tensor.cat(*ws, dim=0).contiguous().realize()
    blk._proj, blk._proj_ends = lin, tuple(ends)
    n += 1
  return n

def _install_transposed_linear() -> None:
  """Route `nn.Linear` through a pre-transposed weight (contiguous [in, out]) when one is present.

  tinygrad's `x.linear(w.transpose())` reads the B operand with a transposed stride; a materialized copy lets
  the Metal tensor-core schedule coalesce it (measured up to 2x on the Kev GEMM shapes)."""
  if getattr(nn.Linear, "_tinykev_t", False): return
  orig = nn.Linear.__call__
  def call(self, x):
    return x.linear(self._weight_t, self.bias) if getattr(self, "_weight_t", None) is not None else orig(self, x)
  nn.Linear.__call__ = call
  nn.Linear._tinykev_t = True

def transpose_linears(model, skip:tuple[str, ...]=("output",)) -> int:
  """Materialize `weight.T` for every Linear in `model` (except `skip`) and free the original."""
  _install_transposed_linear()
  seen:set[int] = set()
  def walk(obj, name:str) -> int:
    if isinstance(obj, Tensor) or id(obj) in seen: return 0
    seen.add(id(obj))
    if isinstance(obj, nn.Linear):
      if not hasattr(obj, "_weight_t"):
        obj._weight_t = obj.weight.T.contiguous().realize()
        if hasattr(obj, "use_custom_quant"): obj.use_custom_quant = False   # tinygrad.llm.model.Linear GGUF path
        return 1
      return 0
    n = 0
    if isinstance(obj, (list, tuple)):
      for v in obj: n += walk(v, name)
    elif hasattr(obj, "__dict__"):
      for k, v in vars(obj).items():
        if k in skip: continue
        n += walk(v, k)
    return n
  return walk(model, "model")

@dataclass
class Loaded:
  model:Transformer
  meta:dict
  tc:dict
  model_dir:pathlib.Path

def load(run:str, max_context:int=8192, dtype:dtypes.DType=dtypes.float16, device:str|None=None,
         lora_scale:float=1.0) -> Loaded:
  """Resolve a Kev checkpoint, build the Qwen3.5 backbone and load merged weights onto `device`."""
  device = device or Tensor.empty(0).device
  run_dir, meta = resolve(run)
  base_dir = resolve_base(meta)
  cfg = json.loads((base_dir / "config.json").read_text())
  tc, prefix = text_config(cfg), BASE_PREFIX
  if meta.get("weights", "lora") == "full":
    # a full checkpoint carries its own text config and shards; the 27B is published this way
    cfg = json.loads((run_dir / "config.json").read_text())
    tc, prefix, base_dir = text_config(cfg), "", run_dir
  model = Transformer(transformer_config(tc, max_context))
  sd = base_state_dict(base_dir, tc, prefix)
  if meta.get("weights", "lora") == "lora":
    if (n := merge_lora(sd, run_dir, tc, device)) == 0: raise ValueError(f"{run_dir} has no LoRA tensors")
  # tie the (unused) LM head to the embeddings so strict loading works without a second 500 MB copy
  sd.setdefault("output.weight", sd["token_embd.weight"])
  sd = {k: (v.to(device).cast(dtype) if dtypes.is_float(v.dtype) and not k.endswith(("ssm_a", "ssm_dt.bias")) else v.to(device))
        for k, v in sd.items()}
  missing = [k for k in nn.state.get_state_dict(model) if k not in sd]
  if missing: raise ValueError(f"missing weights: {missing[:8]}{'...' if len(missing) > 8 else ''}")
  nn.state.load_state_dict(model, sd, verbose=False, consume=True, realize=False)
  Tensor.realize(*nn.state.get_parameters(model))
  del sd
  if os.environ.get("TINYKEV_WT", "0") != "0": transpose_linears(model)
  if os.environ.get("TINYKEV_FUSE_MLP", "0") != "0": fuse_mlp(model)
  if os.environ.get("TINYKEV_FUSE_DELTA", "0") != "0": fuse_delta_proj(model)
  return Loaded(model, meta, tc, run_dir)
