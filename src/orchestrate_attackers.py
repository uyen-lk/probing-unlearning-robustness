#!/usr/bin/env python3
"""
Run multiple attacker scripts and merge their outputs into a single CSV.

"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

# --------- Configure how to run each attacker ----------
# Edit the 'cmd' to match your actual file location and CLI flags.
# {py} -> python executable
# {script} -> path to attacker script
# {inp} -> input CSV path
# {out} -> output CSV path
ATTACKERS = {
    "DAN": {
        "script_candidates": ["attackers/DAN.py", "DAN.py"],
        "cmd": "{py} {script} --input {inp} --output {out}",
        "defaults": {"strategy": "direct"},
    },
    "pretending": {
        "script_candidates": ["attackers/pretending.py", "pretending.py"],
        "cmd": "{py} {script} --input {inp} --output {out}",
        "defaults": {"strategy": "direct"},
    },
    "refusal_suppression": {
        "script_candidates": ["attackers/refusal_suppression.py", "refusal_suppression.py"],
        "cmd": "{py} {script} --input {inp} --output {out}",
        "defaults": {"strategy": "direct"},
    },
    "prefix_injection": {
        "script_candidates": ["attackers/prefix_injection.py", "prefix_injection.py"],
        "cmd": "{py} {script} --input {inp} --output {out}",
        "defaults": {"strategy": "direct"},
    },
    "step_jailbreaking": {
        "script_candidates": ["attackers/step_jailbreaking.py", "step_jailbreaking.py"],
        "cmd": "{py} {script} --input {inp} --output {out}",
        "defaults": {"strategy": "direct"},
    },
}

# Columns we normalize toward
TARGET_COLS = ["base_question", "base_answer", "attack_type", "strategy", "attack_prompt"]

# Flexible renaming map from various attacker outputs -> unified schema
RENAME_MAP = {
    # question
    "question": "base_question",
    "base_question": "base_question",
    "orig_question": "base_question",
    # answer
    "answer": "base_answer",
    "base_answer": "base_answer",
    # prompt
    "prompt": "attack_prompt",
    "attack_prompt": "attack_prompt",
    "model_generated": "attack_prompt",
    # type/strategy
    "type": "attack_type",
    "attack_type": "attack_type",
    "strategy": "strategy",
}


def resolve_script_path(candidates):
    """Pick the first existing path from candidates."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    # If none found, return the first candidate as a best guess (user can fix)
    return candidates[0]


def run_attacker(attacker_name, input_csv, tmp_dir, python_exec="python"):
    meta = ATTACKERS[attacker_name]
    script = resolve_script_path(meta["script_candidates"])
    out_csv = str(Path(tmp_dir) / f"{attacker_name}.csv")
    cmd = meta["cmd"].format(py=python_exec, script=script, inp=input_csv, out=out_csv)

    print(f"[INFO] Running {attacker_name}: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {attacker_name} failed.\nSTDOUT:\n{e.stdout.decode(errors='ignore')}\nSTDERR:\n{e.stderr.decode(errors='ignore')}",
              file=sys.stderr)
        # Continue on error; we’ll just skip this attacker’s output
        return None

    return out_csv


def normalize_columns(df, attacker_name, defaults):
    # 1) Rename columns to our target names when possible
    renamer = {c: RENAME_MAP[c] for c in df.columns if c in RENAME_MAP}
    df = df.rename(columns=renamer)

    # 2) Ensure required columns exist
    if "attack_type" not in df.columns:
        df["attack_type"] = attacker_name
    if "strategy" not in df.columns:
        df["strategy"] = defaults.get("strategy", "direct")
    if "base_question" not in df.columns and "question" in df.columns:
        df["base_question"] = df["question"]
    if "base_answer" not in df.columns and "answer" in df.columns:
        df["base_answer"] = df["answer"]
    if "attack_prompt" not in df.columns and "prompt" in df.columns:
        df["attack_prompt"] = df["prompt"]

    # 3) Keep all extra columns, but make sure target columns are present
    for col in ["base_question", "base_answer", "attack_type", "strategy", "attack_prompt"]:
        if col not in df.columns:
            df[col] = ""

    # 4) Trim whitespace in key text fields
    for col in ["base_question", "base_answer", "attack_prompt"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def merge_csvs(paths):
    dfs = []
    for p in paths:
        if p is None:
            continue
        f = Path(p)
        if not f.exists():
            print(f"[WARN] Skipping missing file: {p}")
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=TARGET_COLS)
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the input CSV with questions/answers.")
    ap.add_argument("--output", required=True, help="Path to the final merged CSV.")
    ap.add_argument("--tmp_dir", default="output/attacked/tmp", help="Where to store intermediate CSVs.")
    ap.add_argument("--attackers", default="DAN,pretending,refusal_suppression,prefix_injection,step_jailbreaking",
                    help="Comma-separated list of attacker names to run.")
    ap.add_argument("--python_exec", default="python", help="Python executable to use.")
    ap.add_argument("--merge_only_dir", default="", help="If provided, skip running attackers and just merge all CSVs in this directory.")
    ap.add_argument("--dedup", action="store_true", help="Drop duplicates by (base_question, attack_type, strategy, attack_prompt).")
    args = ap.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only_dir:
        print(f"[INFO] Merge-only mode: reading all CSVs in {args.merge_only_dir}")
        all_parts = sorted(Path(args.merge_only_dir).glob("*.csv"))
        merged = merge_csvs([str(p) for p in all_parts])
    else:
        attacker_list = [a.strip() for a in args.attackers.split(",") if a.strip()]
        produced_paths = []
        for name in attacker_list:
            if name not in ATTACKERS:
                print(f"[WARN] Unknown attacker '{name}'. Known: {list(ATTACKERS.keys())}")
                continue
            out_path = run_attacker(name, args.input, tmp_dir, python_exec=args.python_exec)
            produced_paths.append(out_path)

        # Load, normalize, and tag by attacker name
        all_norm = []
        for name, csv_path in zip(attacker_list, produced_paths):
            if csv_path is None:
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[WARN] Could not read {csv_path}: {e}")
                continue
            df = normalize_columns(df, attacker_name=name, defaults=ATTACKERS[name].get("defaults", {}))
            all_norm.append(df)

        if not all_norm:
            print("[ERROR] No attacker outputs were produced. Exiting.")
            sys.exit(2)

        merged = pd.concat(all_norm, ignore_index=True)

    # Optional dedup
    if args.dedup:
        merged = merged.drop_duplicates(subset=["base_question", "attack_type", "strategy", "attack_prompt"])

    # Reorder to put target cols first, keep the rest after
    front = [c for c in TARGET_COLS if c in merged.columns]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[DONE] Merged CSV written to: {out_path}")


if __name__ == "__main__":
    main()
