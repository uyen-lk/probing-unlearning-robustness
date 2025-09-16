#!/usr/bin/env python3
"""
Combine many CSVs into one CSV (union of columns, optional de-dup, with provenance).

Examples
--------
# Combine all model generations saved as outdir/<model_tag>/<model_tag>_gen.csv
python src/combine_csv.py --indir output --pattern "*_gen.csv" --out_csv output/all_inference.csv \
  --dedup_by auto


"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Combine multiple CSV files into one.")
    ap.add_argument("--indir", required=True, type=str, help="Root directory to search for CSVs.")
    ap.add_argument("--out_csv", required=True, type=str, help="Path to write the combined CSV.")
    ap.add_argument("--pattern", type=str, default="*.csv", help="Filename glob (default: *.csv).")
    ap.add_argument("--recursive", type=lambda v: str(v).lower() in {"1","true","t","yes","y"},
                    default=True, help="Recurse into subdirectories (default: true).")
    ap.add_argument("--encoding", type=str, default="utf-8",
                    help="Preferred input encoding (default: utf-8).")
    ap.add_argument("--fallback_encodings", type=str, default="utf-8-sig,latin1",
                    help="Comma-separated fallback encodings to try on read failure.")
    ap.add_argument("--dedup_by", type=str, default="",
                    help="Comma-separated columns to drop duplicates on; "
                         "'auto' = all non-provenance columns; empty = no de-dup.")
    ap.add_argument("--sort_cols", action="store_true",
                    help="Sort columns alphabetically in the final CSV (after any --columns ordering).")
    ap.add_argument("--sort_by", type=str, default="",
                    help="Comma-separated columns to sort rows by before saving.")
    ap.add_argument("--columns", type=str, default="",
                    help="Comma-separated list of output columns in desired order.")
    ap.add_argument("--keep_only", action="store_true",
                    help="Keep ONLY the columns listed in --columns (drop all others).")
    ap.add_argument("--keep_first", action="store_true",
                    help="If --columns is empty, keep the column order of the first readable CSV.")
    ap.add_argument("--add_provenance", action="store_true",
                    help="Add provenance columns: source_path, source_dir, source_file, model_tag.")
    return ap.parse_args()


def find_csvs(indir: Path, pattern: str, recursive: bool) -> List[Path]:
    return sorted(indir.rglob(pattern) if recursive else indir.glob(pattern))


def read_csv_robust(path: Path, enc: str, fallbacks: List[str]) -> pd.DataFrame:
    encodings = [enc] + [e.strip() for e in fallbacks if e.strip()]
    last_err: Optional[Exception] = None
    for e in encodings:
        try:
            return pd.read_csv(path, encoding=e, low_memory=False)
        except Exception as ex:
            last_err = ex
    raise RuntimeError(f"Failed to read CSV '{path}' with tried encodings {encodings}: {last_err}")


def enforce_columns(df: pd.DataFrame, ordered_cols: List[str], keep_only: bool) -> pd.DataFrame:
    # Ensure requested columns exist
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = pd.NA
    if keep_only:
        return df[ordered_cols]
    # Otherwise, place requested columns first, then any others
    others = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + others]


def main():
    args = parse_args()
    root = Path(args.indir)
    if not root.exists():
        raise FileNotFoundError(f"--indir does not exist: {root}")

    fallbacks = args.fallback_encodings.split(",") if args.fallback_encodings else []

    csv_paths = find_csvs(root, args.pattern, args.recursive)
    if not csv_paths:
        raise SystemExit(f"No CSV files matched pattern '{args.pattern}' under {root}")

    dfs = []
    first_cols: Optional[List[str]] = None

    for p in tqdm(csv_paths, desc="Reading CSVs"):
        try:
            df = read_csv_robust(p, args.encoding, fallbacks)
        except Exception as e:
            print(f"[WARN] Skipping {p} due to read error: {e}")
            continue

        if args.add_provenance:
            df = df.copy()
            df["source_path"] = str(p)
            df["source_dir"] = str(p.parent)
            df["source_file"] = p.name
            df["model_tag"] = p.parent.name  # outdir/<model_tag>/<file>.csv

        if first_cols is None:
            first_cols = list(df.columns)

        dfs.append(df)

    if not dfs:
        raise SystemExit("No CSVs were read successfully; nothing to combine.")

    combined = pd.concat(dfs, axis=0, ignore_index=True, sort=True)

    # Determine desired column order
    desired_cols: List[str] = []
    if args.columns.strip():
        desired_cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    elif args.keep_first and first_cols is not None:
        desired_cols = first_cols

    if desired_cols:
        combined = enforce_columns(combined, desired_cols, args.keep_only)

    # Optional: sort columns alphabetically (but keep requested order in front if provided)
    if args.sort_cols and not args.keep_only:
        if desired_cols:
            front = [c for c in combined.columns if c in desired_cols]
            rest = sorted([c for c in combined.columns if c not in desired_cols])
            combined = combined[front + rest]
        else:
            combined = combined[sorted(combined.columns)]

    # De-dup
    dd = args.dedup_by.strip()
    prov_cols = {"source_path", "source_dir", "source_file", "model_tag"}
    if dd:
        if dd.lower() == "auto":
            subset = [c for c in combined.columns if c not in prov_cols]
            if subset:
                before = len(combined)
                combined = combined.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
                print(f"[INFO] Dedup(auto): {before} -> {len(combined)} rows (subset={len(subset)} cols)")
        else:
            subset = [c.strip() for c in dd.split(",") if c.strip() in combined.columns]
            if subset:
                before = len(combined)
                combined = combined.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
                print(f"[INFO] Dedup({subset}): {before} -> {len(combined)} rows")
            else:
                print(f"[WARN] --dedup_by specified but no matching columns found; skipping de-dup.")

    # Optional: sort rows
    if args.sort_by.strip():
        by = [c.strip() for c in args.sort_by.split(",") if c.strip() in combined.columns]
        if by:
            combined = combined.sort_values(by=by, kind="mergesort").reset_index(drop=True)
        else:
            print(f"[WARN] --sort_by had no valid columns; skipping row sort.")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"[OK] Wrote combined CSV -> {out_path}")
    print(f"[OK] Rows: {len(combined)}, Cols: {len(combined.columns)}")


if __name__ == "__main__":
    main()
