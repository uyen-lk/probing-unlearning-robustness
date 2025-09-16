#!/usr/bin/env python3
"""
Fully-configurable generator for Hugging Face (unlearned) models over an attack-prompt CSV.

"""

import os
import re
import json
import math
import argparse
from typing import Iterable, List, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from huggingface_hub import login

import string

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}

def maybe_hf_login_from_env() -> None:
    # Tip: commonly-used env var is HUGGINGFACE_HUB_TOKEN
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        try:
            login(token=token)
        except Exception as e:
            print(f"[WARN] Could not login to Hugging Face with env token: {e}")

def choose_torch_dtype(dtype_arg: str) -> torch.dtype:
    """
    Map dtype_arg to a torch dtype.
    - 'auto': bf16 if supported on CUDA; else fp16 on CUDA; else fp32
    - 'bf16' | 'fp16' | 'fp32' accepted
    """
    dtype_arg = (dtype_arg or "auto").lower()
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "fp32":
        return torch.float32

    # auto
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        except Exception:
            return torch.float16
    return torch.float32

def format_prompt(prompt: str, tokenizer, use_chat_template: bool) -> str:
    p = (prompt or "").strip()
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": p}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return p
    return p

def batched(seq: List, n: int) -> Iterable[List]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def clean_response(text: str) -> str:
    """Remove leading non-alphanumeric symbols from model output."""
    if not text:
        return text
    t = text.strip()
    allowed = string.ascii_letters + string.digits
    i = 0
    while i < len(t) and t[i] not in allowed:
        i += 1
    return t[i:].lstrip()

# -----------------------------
# Parsing helpers for backbone/method
# -----------------------------
_BACKBONES = [
    "llama", "phi"
]
_ALGOS_CANON = [
    "GA", "GD", "DPO", "NPO", "RMU", "UNDIAL", "SIMNPO", "LAU", "LAT", "RMAT"
]
_ALGO_SET_UPPER = {a.upper() for a in _ALGOS_CANON}
_BACKBONE_SET_LOWER = set(_BACKBONES)

def _split_tokens(name: str) -> List[str]:
    # split by any non-alphanumeric (underscore, dash, etc.)
    return [t for t in re.split(r"[^A-Za-z0-9]+", name) if t]

def _find_backbone(tokens: List[str]) -> Optional[str]:
    for tok in tokens:
        low = tok.lower()
        for bb in _BACKBONES:
            if low.startswith(bb):
                return tok
    return None

def _find_algo(tokens: List[str]) -> Optional[str]:
    # Prefer early tokens
    for tok in tokens:
        if tok.upper() in _ALGO_SET_UPPER:
            # Preserve original casing
            return tok
    return None

def parse_backbone_and_model(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristics:
      - look at repo tag, split on non-alphanumerics
      - detect backbone (phi/llama/...) and algo (GA/GD/DPO/NPO/...)
    """
    tag = model_id.split("/")[-1]
    tokens = _split_tokens(tag)

    backbone = None
    algo = None

    if tokens:
        if tokens[0].lower() in _BACKBONE_SET_LOWER:
            backbone = tokens[0]
            if len(tokens) > 1 and tokens[1].upper() in _ALGO_SET_UPPER:
                algo = tokens[1]
        elif tokens[0].upper() in _ALGO_SET_UPPER:
            algo = tokens[0]

    backbone = backbone or _find_backbone(tokens)
    algo = algo or _find_algo(tokens)

    return backbone, algo

def build_generation_config(tokenizer, args) -> GenerationConfig:
    cfg = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.do_sample:
        cfg.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        })
    else:
        cfg.update({
            "do_sample": False,
        })
    return GenerationConfig(**cfg)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Flexible text generation for attack prompts.")
    ap.add_argument("--model", required=True, type=str, help="Hugging Face model ID.")
    ap.add_argument("--input", required=True, type=str, help="Input CSV with prompts (default column 'attack_prompt').")
    ap.add_argument("--output", required=True, type=str, help="Output directory to save CSV/JSONL.")
    ap.add_argument("--prompt_col", type=str, default="attack_prompt", help="Column name that contains the prompt text.")

    # decoding
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", type=str2bool, default=False, help="true/false")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)

    # system/perf
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    # formatting
    ap.add_argument("--use_chat_template", type=str2bool, default=True, help="true/false")
    ap.add_argument("--padding_side", type=str, default="left", choices=["left", "right", "auto"])

    return ap.parse_args()

def main():
    args = parse_args()
    maybe_hf_login_from_env()

    # read data
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    df = pd.read_csv(args.input)
    if args.prompt_col not in df.columns:
        raise ValueError(f"Column '{args.prompt_col}' not found. Available: {list(df.columns)}")

    # model + tokenizer
    torch_dtype = choose_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # padding side
    if args.padding_side == "auto":
        try:
            tokenizer.padding_side = "left"  # default for decoder-only
        except Exception:
            pass
    else:
        tokenizer.padding_side = args.padding_side

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    # parse backbone & algo from model id
    backbone, algo = parse_backbone_and_model(args.model)

    # generation config & seed
    gen_cfg = build_generation_config(tokenizer, args)
    torch.manual_seed(args.seed)

    # prepare data
    os.makedirs(args.output, exist_ok=True)
    rows = list(df.itertuples(index=False))
    total_batches = math.ceil(len(rows) / args.batch_size)

    records = []

    # batched generation
    for batch in tqdm(list(batched(rows, args.batch_size)), total=total_batches, desc="Generating"):
        prompts = [format_prompt(getattr(r, args.prompt_col), tokenizer, args.use_chat_template) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            out = model.generate(**enc, generation_config=gen_cfg)

        # Only the generated continuation
        gen_ids = out[:, enc.input_ids.shape[1]:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for r, resp in zip(batch, texts):
            rec = r._asdict() if hasattr(r, "_asdict") else dict(r._mapping)
            rec.update({
                "model_name": args.model,      # full HF ID
                "backbone": backbone,          # e.g., phi / Llama / etc.
                "model": algo,                 # e.g., GA / GD / DPO / NPO / ...
                "model_response": clean_response(resp),
            })
            records.append(rec)

    # save
    model_tag = args.model.split("/")[-1]  # short name for folder
    model_outdir = os.path.join(args.output, model_tag)
    os.makedirs(model_outdir, exist_ok=True)

    csv_path = os.path.join(model_outdir, f"{model_tag}_gen.csv")
    jsonl_path = os.path.join(model_outdir, f"{model_tag}_gen.jsonl")

    pd.DataFrame(records).to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Parsed backbone={backbone} model={algo} from '{args.model}'")
    print(f"[OK] Saved CSV   -> {csv_path}")
    print(f"[OK] Saved JSONL -> {jsonl_path}")

if __name__ == "__main__":
    main()
