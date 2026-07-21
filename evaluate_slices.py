#!/usr/bin/env python3
"""Per-cohort weakness analysis for the text-only seqrec encoder.

Computes per-example rank/recall/ndcg on the test split, then aggregates by
user-history length, target-item train-frequency quintile, novel-vs-repeat
target, and rank buckets.
"""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from data import (
    DATASET_REGISTRY,
    Example,
    HistoryFormatConfig,
    build_eval_catalog,
    build_examples,
    build_user_sequences,
    load_interactions,
    load_item_texts,
)
from ranking import per_example_ranks  # shared rank path; do not reimplement locally
from train_fusion import encode_texts_cached, encode_with_dedup, encoder_cache_salt
from utils import save_json, set_seed
import cohort_dims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqrec-checkpoint", type=str,
                   default="outputs/seqrec_qwen3_beauty/best")
    p.add_argument("--dataset-name", type=str,
                   choices=sorted(DATASET_REGISTRY.keys()), default="Beauty")
    p.add_argument("--root", type=str, default="./raw_data")
    p.add_argument("--rating-score", type=float, default=0.0)
    p.add_argument("--download-if-missing", action="store_true", default=True)
    p.add_argument("--max-title-words", type=int, default=20)
    p.add_argument("--max-history-items", type=int, default=20)
    p.add_argument("--min-user-seq-len", type=int, default=3)
    p.add_argument("--history-time-text", action="store_true")
    p.add_argument("--history-rating-text", action="store_true")
    p.add_argument("--history-sep", action="store_true")
    p.add_argument("--history-pos-marker", action="store_true")
    p.add_argument("--enrich-text-input", action="store_true",
                   help="Enable all history enrichment flags (matches qwen3 training).")
    p.add_argument("--eval-catalog", type=str,
                   choices=["interacted", "metadata"], default="interacted")
    p.add_argument("--encode-batch-size", type=int, default=8)
    p.add_argument("--doc-chunk-size", type=int, default=4096,
                   help="Query-chunk size for ranking (kept name for shell compat).")
    p.add_argument("--filter-seen-items", action="store_true",
                   help="Mask items in the user's history before ranking (matches train_fusion.py).")
    p.add_argument("--split", type=str, choices=["test", "val"], default="test")
    p.add_argument("--emb-cache-dir", type=str, default="",
                   help="Sharded embedding cache dir (share with train_fusion, "
                        "e.g. outputs/emb_cache_<tag>). Empty = no caching.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to write per_example.parquet + slice_metrics.json. "
                        "Default: <seqrec_checkpoint>/../slices")
    return p.parse_args()


# ---------- cohort aggregation ----------

def _agg(mask: np.ndarray, rank: np.ndarray) -> Dict[str, float]:
    n = int(mask.sum())
    if n == 0:
        return {"count": 0, "recall@5": 0.0, "recall@10": 0.0,
                "ndcg@10": 0.0, "mrr": 0.0}
    r = rank[mask]
    hit5 = (r <= 5).astype(np.float32)
    hit10 = (r <= 10).astype(np.float32)
    ndcg10 = np.where(r <= 10, 1.0 / np.log2(r.astype(np.float64) + 1.0), 0.0)
    mrr = 1.0 / r.astype(np.float64)
    return {
        "count": n,
        "recall@5": float(hit5.mean()),
        "recall@10": float(hit10.mean()),
        "ndcg@10": float(ndcg10.mean()),
        "mrr": float(mrr.mean()),
    }


def _bin_history_len(hl: np.ndarray) -> Tuple[List[str], List[np.ndarray]]:
    bins = [
        ("1", hl == 1),
        ("2", hl == 2),
        ("3-5", (hl >= 3) & (hl <= 5)),
        ("6-10", (hl >= 6) & (hl <= 10)),
        ("11-20", (hl >= 11) & (hl <= 20)),
        ("21+", hl >= 21),
    ]
    return [b[0] for b in bins], [b[1] for b in bins]


def _bin_popularity(freq: np.ndarray) -> Tuple[List[str], List[np.ndarray], List[float]]:
    qs = np.quantile(freq, [0.2, 0.4, 0.6, 0.8])
    e = [float(q) for q in qs]
    labels = [
        f"Q1 (freq<={e[0]:.0f})",
        f"Q2 ({e[0]:.0f}<freq<={e[1]:.0f})",
        f"Q3 ({e[1]:.0f}<freq<={e[2]:.0f})",
        f"Q4 ({e[2]:.0f}<freq<={e[3]:.0f})",
        f"Q5 (freq>{e[3]:.0f})",
    ]
    masks = [
        freq <= e[0],
        (freq > e[0]) & (freq <= e[1]),
        (freq > e[1]) & (freq <= e[2]),
        (freq > e[2]) & (freq <= e[3]),
        freq > e[3],
    ]
    return labels, masks, e


def _rank_buckets(rank: np.ndarray) -> Dict[str, int]:
    return {
        "1-10": int((rank <= 10).sum()),
        "11-100": int(((rank > 10) & (rank <= 100)).sum()),
        "101-1000": int(((rank > 100) & (rank <= 1000)).sum()),
        "1001+": int((rank > 1000).sum()),
    }


def _print_table(title: str, rows: List[Tuple[str, Dict[str, float]]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'cohort':<28} {'n':>7} {'R@5':>8} {'R@10':>8} {'NDCG@10':>9} {'MRR':>8}")
    for label, m in rows:
        print(f"{label:<28} {m['count']:>7d} "
              f"{m['recall@5']:>8.4f} {m['recall@10']:>8.4f} "
              f"{m['ndcg@10']:>9.4f} {m['mrr']:>8.4f}")


# ---------- main ----------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.enrich_text_input:
        args.history_sep = True
        args.history_time_text = True
        args.history_rating_text = True
        args.history_pos_marker = True
    fmt = HistoryFormatConfig(
        time_text=args.history_time_text,
        rating_text=args.history_rating_text,
        sep=args.history_sep,
        pos_marker=args.history_pos_marker,
    )

    item_to_text = load_item_texts(
        dataset_name=args.dataset_name,
        root=args.root,
        download_if_missing=args.download_if_missing,
        max_title_words=args.max_title_words,
    )
    interactions = load_interactions(
        dataset_name=args.dataset_name,
        root=args.root,
        download_if_missing=args.download_if_missing,
        rating_score=args.rating_score,
        valid_item_ids=set(item_to_text.keys()),
    )
    user_sequences = build_user_sequences(interactions)
    _train, val_examples, test_examples = build_examples(
        user_sequences=user_sequences,
        item_to_text=item_to_text,
        min_user_seq_len=args.min_user_seq_len,
        max_history_items=args.max_history_items,
        fmt=fmt,
    )
    test_examples = val_examples if args.split == "val" else test_examples

    eval_item_to_text = build_eval_catalog(
        item_to_text=item_to_text,
        interactions=interactions,
        eval_catalog=args.eval_catalog,
    )
    item_ids = sorted(eval_item_to_text.keys())
    id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    catalog_texts = [eval_item_to_text[i] for i in item_ids]

    # Item frequency over the full interaction log. Each test example removes
    # exactly one occurrence (the held-out target), so for the target item the
    # "train" frequency = total_freq - 1.
    item_freq_total = Counter(x.item_id for x in interactions)

    print(f"Loading encoder from {args.seqrec_checkpoint}")
    model = SentenceTransformer(args.seqrec_checkpoint)
    model.eval()

    cache_salt = encoder_cache_salt(args.seqrec_checkpoint) if args.emb_cache_dir else ""
    if args.emb_cache_dir:
        print(f"Embedding cache dir: {args.emb_cache_dir} (salt={cache_salt})")

    print(f"Encoding catalog ({len(item_ids):,} items)...")
    corpus_emb = encode_texts_cached(
        model, catalog_texts, args.encode_batch_size, is_query=False,
        cache_dir=args.emb_cache_dir, cache_name="catalog", cache_salt=cache_salt,
    )
    print(f"Encoding queries ({len(test_examples):,} {args.split} examples)...")
    queries = [ex.history_text for ex in test_examples]
    query_emb = encode_with_dedup(
        model, queries, args.encode_batch_size, is_query=True,
        cache_dir=args.emb_cache_dir, cache_name=f"{args.split}_query",
        cache_salt=cache_salt,
    )

    gold_indices = np.array([id_to_idx[ex.target_item_id] for ex in test_examples],
                            dtype=np.int64)

    seen_indices: Optional[List[np.ndarray]] = None
    if args.filter_seen_items:
        seen_indices = []
        for ex in test_examples:
            sidx = sorted({id_to_idx[i] for i in ex.history_item_ids if i in id_to_idx})
            seen_indices.append(np.asarray(sidx, dtype=np.int64))
        print(f"filter_seen_items=True; mean seen-items per query: "
              f"{np.mean([len(s) for s in seen_indices]):.2f}")

    print("Computing per-example ranks...")
    rank = per_example_ranks(query_emb, corpus_emb, gold_indices,
                             seen_indices=seen_indices)

    history_len = np.array([len(ex.history_item_ids) for ex in test_examples],
                           dtype=np.int64)
    target_freq = np.array(
        [max(0, item_freq_total[ex.target_item_id] - 1) for ex in test_examples],
        dtype=np.int64,
    )
    target_in_history = np.array(
        [ex.target_item_id in ex.history_item_ids for ex in test_examples],
        dtype=bool,
    )

    df = pd.DataFrame({
        "user_id": [ex.user_id for ex in test_examples],
        "target_item_id": [ex.target_item_id for ex in test_examples],
        "history_len": history_len,
        "target_freq": target_freq,
        "target_in_history": target_in_history,
        "rank": rank,
        "hit@5": rank <= 5,
        "hit@10": rank <= 10,
        "ndcg@10": np.where(rank <= 10, 1.0 / np.log2(rank.astype(np.float64) + 1.0), 0.0),
    })
    # extra cohort axes baked into the baseline parquet too (see cohort_dims).
    df = cohort_dims.add_activity_age_columns(df, interactions)

    # ---------- aggregate ----------
    overall = _agg(np.ones(len(rank), dtype=bool), rank)

    hl_labels, hl_masks = _bin_history_len(history_len)
    hl_rows = [(lab, _agg(m, rank)) for lab, m in zip(hl_labels, hl_masks)]

    pop_labels, pop_masks, pop_edges = _bin_popularity(target_freq)
    pop_rows = [(lab, _agg(m, rank)) for lab, m in zip(pop_labels, pop_masks)]

    nov_rows = [
        ("novel (not in history)", _agg(~target_in_history, rank)),
        ("repeat (in history)", _agg(target_in_history, rank)),
    ]

    rank_global = _rank_buckets(rank)
    rank_by_hl = {lab: _rank_buckets(rank[m]) for lab, m in zip(hl_labels, hl_masks)}

    _print_table("Overall", [("ALL", overall)])
    _print_table("By user history length", hl_rows)
    _print_table(f"By target popularity (train freq, edges={pop_edges})", pop_rows)
    _print_table("Novel vs repeat target", nov_rows)

    print("\n=== Rank buckets (global) ===")
    total = len(rank)
    for lab, n in rank_global.items():
        print(f"  {lab:<10} {n:>7d}  ({100.0 * n / total:5.2f}%)")
    print("\n=== Rank buckets by history length ===")
    print(f"{'cohort':<10} {'1-10':>8} {'11-100':>8} {'101-1k':>8} {'1k+':>8}")
    for lab, buckets in rank_by_hl.items():
        print(f"{lab:<10} {buckets['1-10']:>8d} {buckets['11-100']:>8d} "
              f"{buckets['101-1000']:>8d} {buckets['1001+']:>8d}")

    # ---------- write outputs ----------
    if args.output_dir is None:
        ckpt_parent = os.path.dirname(os.path.abspath(args.seqrec_checkpoint.rstrip("/")))
        out_dir = os.path.join(ckpt_parent, "slices")
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    parquet_path = os.path.join(out_dir, "per_example.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"\nWrote per-example results -> {parquet_path}")

    metrics_path = os.path.join(out_dir, "slice_metrics.json")
    payload = {
        "seqrec_checkpoint": args.seqrec_checkpoint,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "filter_seen_items": bool(args.filter_seen_items),
        "num_examples": int(len(rank)),
        "num_corpus_items": int(len(item_ids)),
        "overall": overall,
        "by_history_len": dict(hl_rows),
        "by_target_popularity": {
            "quintile_edges": pop_edges,
            "cohorts": dict(pop_rows),
        },
        "by_novel_vs_repeat": dict(nov_rows),
        "rank_buckets_global": rank_global,
        "rank_buckets_by_history_len": rank_by_hl,
    }
    save_json(metrics_path, payload)
    print(f"Wrote slice metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
