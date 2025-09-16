#!/usr/bin/env python3
"""
Evaluate text-similarity & leakage metrics for unlearning analysis.


Usage:
  python eval.py --input in.csv --output out.csv
  # Optional knobs:
  python eval.py --input in.csv --output out.csv \
      --ref_col base_answer --hyp_col model_response \
      --sbert_model sentence-transformers/all-mpnet-base-v2 \
      --use_model sentence-transformers/distiluse-base-multilingual-cased-v1 \
      --bertscore-model microsoft/deberta-xlarge-mnli \
      --bertscore-lang en \
      --bertscore-batch-size 64

Notes:
  • SBERT and USE are batched for speed (single forward pass each).
  • Empty/NaN pairs are scored as 0.0 across all metrics.
  • USE is loaded via SentenceTransformer() (e.g., "universal-sentence-encoder" or any ST checkpoint).
  • BERTScore runs on CPU or CUDA; with rescaling by language baseline for comparability.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch

# Quiet some libs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_grad_enabled(False)

# ---- NLTK setup (download once) ----
import nltk
for pkg in ("punkt", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# ---- Other metrics ----
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# ---- BERTScore (optional dependency) ----
_BERTSCORE_AVAILABLE = True
try:
    from bert_score import score as bertscore_score
except Exception:
    _BERTSCORE_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute similarity metrics for unlearning evaluation.")
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", required=True, help="Path to write augmented CSV")
    ap.add_argument("--ref_col", default="base_answer", help="Reference text column (default: base_answer)")
    ap.add_argument("--hyp_col", default="model_response", help="Hypothesis text column (default: model_response)")
    ap.add_argument("--sbert_model", default="sentence-transformers/all-mpnet-base-v2",
                    help="SentenceTransformer model for SBERT similarity")
    ap.add_argument("--use_model", default="sentence-transformers/distiluse-base-multilingual-cased-v1",
                    help="SentenceTransformer model name for USECS (e.g., 'universal-sentence-encoder')")
    ap.add_argument("--bertscore-model", dest="bertscore_model", default="microsoft/deberta-xlarge-mnli",
                    help="BERTScore backbone (default: microsoft/deberta-xlarge-mnli)")
    ap.add_argument("--bertscore-lang", dest="bertscore_lang", default="en",
                    help="Language code for BERTScore baseline rescaling (default: en)")
    ap.add_argument("--bertscore-batch-size", dest="bertscore_batch_size", type=int, default=64,
                    help="Batch size for BERTScore (default: 64)")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding models")
    return ap.parse_args()


def _clean_series(s: pd.Series) -> pd.Series:
    """
    Coerce to string, strip whitespace, and treat '', 'nan', 'null', 'none' (any case) as empty.
    Works on older pandas (no 'case=' kw on Series.eq).
    """
    s = s.astype(str).str.strip()
    s_lower = s.str.lower()
    bad = s_lower.isin(["", "nan", "null", "none"])
    s = s.mask(bad, "").fillna("")
    return s


def _cosine_diag(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Diagonal cosine similarities between matching rows of a and b."""
    sim = util.cos_sim(a, b)  # [N, N]
    return sim.diag().detach().cpu().numpy()


def compute_metrics(
    refs: List[str],
    hyps: List[str],
    sbert_model_name: str,
    use_model_name: str,
    batch_size: int,
    device: str,
    berts_model_name: str,
    berts_lang: str,
    berts_batch_size: int,
) -> pd.DataFrame:
    n = len(refs)
    assert n == len(hyps)

    # Valid pairs mask (both non-empty)
    valid = np.array([(bool(r) and bool(h)) for r, h in zip(refs, hyps)], dtype=bool)

    # ----- ROUGE-L (recall) -----
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = np.zeros(n, dtype=float)
    for i in range(n):
        if valid[i]:
            scores = rouge.score(refs[i], hyps[i])
            rougeL[i] = scores["rougeL"].recall
        else:
            rougeL[i] = 0.0

    # ----- SBERT cosine (batched) -----
    sbert = SentenceTransformer(sbert_model_name, device=device)
    sbert_ref = sbert.encode(refs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    sbert_hyp = sbert.encode(hyps, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    SBERT_sim = _cosine_diag(sbert_ref, sbert_hyp)
    SBERT_sim[~valid] = 0.0

    # ----- USE cosine (batched) -----
    try:
        use_model = SentenceTransformer(use_model_name, device=device)
        use_ref = use_model.encode(refs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        use_hyp = use_model.encode(hyps, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        USECS_sim = _cosine_diag(use_ref, use_hyp)
        USECS_sim[~valid] = 0.0
    except Exception as e:
        print(f"[WARN] Failed to load USE model '{use_model_name}': {e}\n"
              f"       Filling usecs_semantic_sim with 0.0. "
              f"       (Install tensorflow-backed USE or pick another ST model.)")
        USECS_sim = np.zeros(n, dtype=float)

    # ----- BLEU + METEOR (tokenized per pair) -----
    smoothie = SmoothingFunction().method4
    bleu = np.zeros(n, dtype=float)
    meteor = np.zeros(n, dtype=float)
    for i in range(n):
        if not valid[i]:
            bleu[i] = 0.0
            meteor[i] = 0.0
            continue
        ref_tok = word_tokenize(refs[i])
        hyp_tok = word_tokenize(hyps[i])
        try:
            bleu[i] = sentence_bleu([ref_tok], hyp_tok, smoothing_function=smoothie)
        except ZeroDivisionError:
            bleu[i] = 0.0
        try:
            meteor[i] = meteor_score([ref_tok], hyp_tok)
        except Exception:
            meteor[i] = 0.0

    # ----- BERTScore (F1, batched) -----
    bertscore_f1 = np.zeros(n, dtype=float)
    if _BERTSCORE_AVAILABLE:
        try:
            # bert-score uses "cuda" or "cpu"
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            # Score expects lists of strings (cands = hyps, refs = refs)
            # rescale_with_baseline=True recommended for cross-model comparability
            _, _, F1 = bertscore_score(
                cands=hyps,
                refs=refs,
                model_type=berts_model_name,
                lang=berts_lang,
                device=device_str,
                batch_size=berts_batch_size,
                rescale_with_baseline=True,
                verbose=False,
            )
            bertscore_f1 = F1.detach().cpu().numpy()
            bertscore_f1[~valid] = 0.0
        except Exception as e:
            print(f"[WARN] BERTScore failed ({berts_model_name}): {e}\n"
                  f"       Filling bertscore_f1 with 0.0.")
            bertscore_f1 = np.zeros(n, dtype=float)
    else:
        print("[WARN] 'bert-score' not installed. Run `pip install bert-score` to enable it.")
        bertscore_f1 = np.zeros(n, dtype=float)

    return pd.DataFrame(
        {
            "rougeL": rougeL,
            "bleu_score": bleu,
            "meteor_score": meteor,
            "bertscore_f1": bertscore_f1,
            "SBERT_semantic_sim": SBERT_sim,
            "usecs_semantic_sim": USECS_sim,
        }
    )


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.ref_col not in df.columns or args.hyp_col not in df.columns:
        raise ValueError(
            f"Missing required columns: '{args.ref_col}' and/or '{args.hyp_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    # Clean text columns
    refs = _clean_series(df[args.ref_col]).tolist()
    hyps = _clean_series(df[args.hyp_col]).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_df = compute_metrics(
        refs=refs,
        hyps=hyps,
        sbert_model_name=args.sbert_model,
        use_model_name=args.use_model,
        batch_size=args.batch_size,
        device=device,
        berts_model_name=args.bertscore_model,
        berts_lang=args.bertscore_lang,
        berts_batch_size=args.bertscore_batch_size,
    )

    out = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)
    out.to_csv(args.output, index=False)
    print(f"[OK] Wrote: {args.output}")
    print("Columns added:", list(metrics_df.columns))


if __name__ == "__main__":
    main()
