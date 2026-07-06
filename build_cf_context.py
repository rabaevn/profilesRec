#!/usr/bin/env python3
"""Build CF context cache keyed by profile_cache_key.

For each unique prefix window (last `max_history` items), emit:
- cooc_titles: items most likely to be bought right after items in this history
  (from train item-item next-item statistics).
- neighbor_titles: items popular among users whose train history overlaps the prefix,
  weighted by overlap size.

Both signals are derived strictly from each user's train portion (sequence minus
the last 2 positions, which are val + test targets) to avoid eval leakage.

Output: JSONL keyed by SHA1 of the last `max_history` item IDs (matches
profiles.profile_cache_key).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
from scipy.sparse import coo_matrix

from data import (
    build_eval_catalog,
    build_user_sequences,
    load_interactions,
    load_item_texts,
)
from profiles import profile_cache_key
from utils import truncate_words


def _clean_title(raw: str, title_words: int) -> str:
    t = raw
    if t.startswith("Item: "):
        t = t[len("Item: "):]
    return truncate_words(t, title_words)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-name", default="Beauty")
    p.add_argument("--root", default="./raw_data")
    p.add_argument("--rating-score", type=float, default=0.0)
    p.add_argument("--max-title-words", type=int, default=20,
                   help="Used to truncate emitted titles in the CF context block.")
    p.add_argument("--max-history", type=int, default=20,
                   help="Must match --llm-profile-max-history used in generate_profiles.py.")
    p.add_argument("--min-user-seq-len", type=int, default=3)
    p.add_argument("--eval-catalog", choices=["interacted", "metadata"], default="interacted")
    p.add_argument("--top-k-cooc", type=int, default=10)
    p.add_argument("--top-k-neighbor", type=int, default=10)
    p.add_argument("--max-neighbor-users", type=int, default=50,
                   help="Cap on the number of nearest users aggregated per prefix.")
    p.add_argument("--output-path", required=True)
    args = p.parse_args()

    item_to_text = load_item_texts(
        dataset_name=args.dataset_name,
        root=args.root,
        download_if_missing=False,
        max_title_words=args.max_title_words,
    )
    interactions = load_interactions(
        dataset_name=args.dataset_name,
        root=args.root,
        download_if_missing=False,
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
    item_idx = {iid: i for i, iid in enumerate(item_ids)}
    num_items = len(item_ids)
    print(f"Catalog: {num_items:,} items  Users: {len(user_sequences):,}")

    # Train portion = all but the last 2 positions (val + test targets stripped).
    # Mirrors the leave-last-two-out boundary in data.build_examples.
    train_seqs: Dict[str, List[int]] = {}
    for uid, seq in user_sequences.items():
        ids = [item_idx[x.item_id] for x in seq if x.item_id in item_idx]
        if len(ids) < 2:
            continue
        train_part = ids[:-2] if len(ids) >= 3 else ids[:-1]
        if train_part:
            train_seqs[uid] = train_part

    # Item-item next-item count matrix M[i, j] = #times j is the immediate next purchase after i.
    t0 = time.time()
    rows: List[int] = []
    cols: List[int] = []
    for ids in train_seqs.values():
        for i, j in zip(ids[:-1], ids[1:]):
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    M_csr = coo_matrix((data, (rows, cols)), shape=(num_items, num_items)).tocsr()
    M_csr.sum_duplicates()
    print(f"next-item matrix: nnz={M_csr.nnz:,} in {time.time()-t0:.1f}s")

    # Inverted index: item -> list of users (train portion only).
    item_users: Dict[int, List[str]] = defaultdict(list)
    user_train_sets: Dict[str, set] = {uid: set(ids) for uid, ids in train_seqs.items()}
    for uid, ids in train_seqs.items():
        for i in user_train_sets[uid]:
            item_users[i].append(uid)
    print(f"inverted index built ({sum(len(v) for v in item_users.values()):,} postings)")

    # Resume if cache file exists.
    seen_keys: set = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen_keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resume: {len(seen_keys):,} keys already in cache")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    out_f = open(args.output_path, "a", encoding="utf-8")

    n_new = 0
    t0 = time.time()
    for uid, seq in user_sequences.items():
        seq_filt = [x for x in seq if x.item_id in item_idx]
        n = len(seq_filt)
        if n < 2:
            continue
        history_id_seq = [item_idx[x.item_id] for x in seq_filt]
        # Match collect_unique_prompts walk: test prefix, val prefix (if eligible), train prefixes.
        prefix_positions: List[int] = [n - 1]
        if n >= args.min_user_seq_len:
            prefix_positions.append(n - 2)
            prefix_positions.extend(range(1, n - 2))

        for pos in prefix_positions:
            prefix_ids_str = [seq_filt[k].item_id for k in range(pos)]
            key = profile_cache_key(prefix_ids_str, args.max_history)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            prefix_idx = history_id_seq[max(0, pos - args.max_history): pos]
            prefix_set = set(prefix_idx)

            # Co-occurrence (next-item) scores summed over prefix.
            sub = M_csr[prefix_idx, :]
            row_sum = np.asarray(sub.sum(axis=0)).ravel()
            for h in prefix_set:
                row_sum[h] = -1.0
            cooc_titles: List[str] = []
            if (row_sum > 0).any():
                k = min(args.top_k_cooc, num_items - 1)
                top = np.argpartition(-row_sum, k)[:k]
                top = top[np.argsort(-row_sum[top])]
                for i in top:
                    if row_sum[i] <= 0:
                        break
                    cooc_titles.append(_clean_title(eval_item_to_text[item_ids[int(i)]], args.max_title_words))

            # Neighbor users: weighted by overlap with prefix.
            user_overlap: Counter = Counter()
            for h in prefix_set:
                for nu in item_users.get(h, ()):
                    if nu != uid:
                        user_overlap[nu] += 1
            neighbor_counts: Counter = Counter()
            for nu, weight in user_overlap.most_common(args.max_neighbor_users):
                for n_item in user_train_sets.get(nu, ()):
                    if n_item not in prefix_set:
                        neighbor_counts[n_item] += weight
            neighbor_titles: List[str] = [
                _clean_title(eval_item_to_text[item_ids[i]], args.max_title_words)
                for i, _c in neighbor_counts.most_common(args.top_k_neighbor)
            ]

            out_f.write(json.dumps({
                "key": key,
                "cooc_titles": cooc_titles,
                "neighbor_titles": neighbor_titles,
            }, ensure_ascii=False) + "\n")
            n_new += 1

            if n_new % 5000 == 0:
                out_f.flush()
                rate = n_new / max(time.time() - t0, 1e-6)
                print(f"  {n_new:,} keys  ({rate:.0f}/s)")

    out_f.close()
    print(f"done: {n_new:,} new entries -> {args.output_path}")


if __name__ == "__main__":
    main()
