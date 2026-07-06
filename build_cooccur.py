#!/usr/bin/env python3
"""Build an item-item co-occurrence matrix from training interactions.

For each user, take the prefix that excludes the last 2 items (val + test gold)
and count every *unordered* pair (i, j) with i != j. Apply min-support filter,
row-normalize so each item's neighborhood is a probability distribution, save as
scipy sparse CSR.

Used by cf_rerank.py to score (history, candidate) pairs by raw co-occurrence
statistics — a signal in a subspace the text encoder does not operate in.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from scipy import sparse

from data import (
    DATASET_REGISTRY,
    build_eval_catalog,
    build_user_sequences,
    load_interactions,
    load_item_texts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-name", type=str,
                   choices=sorted(DATASET_REGISTRY.keys()), default="Beauty")
    p.add_argument("--root", type=str, default="./raw_data")
    p.add_argument("--rating-score", type=float, default=0.0)
    p.add_argument("--download-if-missing", action="store_true", default=True)
    p.add_argument("--max-title-words", type=int, default=20)
    p.add_argument("--eval-catalog", type=str,
                   choices=["interacted", "metadata"], default="interacted")
    p.add_argument("--min-cooccur", type=int, default=2,
                   help="Drop pairs with count < this (denoise long-tail).")
    p.add_argument("--exclude-last-n", type=int, default=2,
                   help="Skip the last N items per user (val+test gold) to prevent leakage.")
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
    eval_item_to_text = build_eval_catalog(
        item_to_text=item_to_text,
        interactions=interactions,
        eval_catalog=args.eval_catalog,
    )
    item_ids = sorted(eval_item_to_text.keys())
    id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    n_items = len(item_ids)
    print(f"n_items = {n_items}")

    counts: dict[tuple[int, int], int] = defaultdict(int)
    n_users = 0
    n_pairs_seen = 0
    excl = args.exclude_last_n
    for user_id, seq in user_sequences.items():
        # Need: at least 2 items in the training prefix to form any pair.
        if len(seq) < excl + 2:
            continue
        train_items = [s.item_id for s in seq[:-excl] if s.item_id in id_to_idx]
        if len(train_items) < 2:
            continue
        n_users += 1
        idx = [id_to_idx[x] for x in train_items]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = idx[i], idx[j]
                counts[(a, b)] += 1
                counts[(b, a)] += 1
                n_pairs_seen += 1
    print(f"counted {n_users} users; {n_pairs_seen:,} unordered pair touches; "
          f"{len(counts):,} unique directed entries before filter")

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for (i, j), v in counts.items():
        if v >= args.min_cooccur:
            rows.append(i)
            cols.append(j)
            vals.append(float(v))
    print(f"after min_cooccur={args.min_cooccur}: {len(vals):,} entries kept")

    M = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32,
    )
    rowsum = np.asarray(M.sum(axis=1)).ravel()
    nonempty = (rowsum > 0).sum()
    print(f"non-empty rows: {nonempty}/{n_items} ({nonempty / n_items:.1%})")
    safe_rowsum = np.where(rowsum > 0, rowsum, 1.0)
    D_inv = sparse.diags(1.0 / safe_rowsum)
    M_norm = (D_inv @ M).tocsr()

    sparse.save_npz(args.output, M_norm)
    print(f"saved {args.output} (nnz={M_norm.nnz:,}, shape={M_norm.shape})")


if __name__ == "__main__":
    main()
