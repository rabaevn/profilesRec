#!/usr/bin/env python3
"""Build per-user CF embeddings from the user-user co-purchase matrix.

Pipeline:
    1. Load interactions and form per-user item sets restricted to the train
       portion (sequence minus the last 2 items, matching build_cf_context.py).
    2. Form sparse user x item binary matrix B (rows L2-normalized).
    3. Form sparse user x user cosine matrix S = B @ B.T, zero the diagonal.
    4. Truncated SVD on S -> U, sigma. User embedding = U @ diag(sqrt(sigma)),
       L2-normalized row-wise.
    5. Save user_ids + embeddings + dim to an .npz.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

from data import (
    build_user_sequences,
    load_interactions,
    load_item_texts,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-name", default="Beauty")
    p.add_argument("--root", default="./raw_data")
    p.add_argument("--rating-score", type=float, default=0.0)
    p.add_argument("--max-title-words", type=int, default=20)
    p.add_argument("--cf-emb-dim", type=int, default=64,
                   help="Truncated-SVD rank = output user embedding dim.")
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

    item_ids_sorted = sorted(item_to_text.keys())
    item_idx = {iid: i for i, iid in enumerate(item_ids_sorted)}
    num_items = len(item_ids_sorted)

    # Train portion per user (mirror build_cf_context.py:87-96).
    # IMPORTANT: keep an entry for every user in user_sequences (even users whose
    # train portion is empty) so the resulting .npz is exhaustive — train_cf_fusion
    # can then look up every val user without silently dropping examples.
    user_train_items: Dict[str, List[int]] = {}
    for uid, seq in user_sequences.items():
        ids = [item_idx[x.item_id] for x in seq if x.item_id in item_idx]
        if len(ids) < 2:
            train_part: List[int] = []
        else:
            train_part = ids[:-2] if len(ids) >= 3 else ids[:-1]
        user_train_items[uid] = sorted(set(train_part))

    user_ids = sorted(user_train_items.keys())
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    num_users = len(user_ids)
    has_cf = np.array(
        [len(user_train_items[uid]) > 0 for uid in user_ids], dtype=bool
    )
    print(
        f"Catalog: {num_items:,} items  Users: {num_users:,} "
        f"(with train portion: {int(has_cf.sum()):,}, empty: {int((~has_cf).sum()):,})"
    )

    # Run SVD on the non-empty users only; users without train interactions get
    # a zero embedding row in the final table (and has_cf=False).
    nonempty_uids = [uid for uid in user_ids if has_cf[user_idx[uid]]]
    nonempty_idx = {uid: i for i, uid in enumerate(nonempty_uids)}
    num_nonempty = len(nonempty_uids)

    rows: List[int] = []
    cols: List[int] = []
    for uid in nonempty_uids:
        u = nonempty_idx[uid]
        for i in user_train_items[uid]:
            rows.append(u)
            cols.append(i)
    data = np.ones(len(rows), dtype=np.float32)
    B = csr_matrix((data, (rows, cols)), shape=(num_nonempty, num_items), dtype=np.float32)
    print(f"user-item matrix (non-empty users only): shape={B.shape} nnz={B.nnz:,}")

    # L2-normalize rows of B.
    row_norms = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
    row_norms[row_norms == 0] = 1.0
    inv_norms = 1.0 / row_norms
    B_norm = B.multiply(inv_norms[:, None]).tocsr().astype(np.float32)

    # Truncated SVD on S = B_norm @ B_norm.T - I.
    #
    # Avoid materializing the user-user matrix: for large datasets (e.g. Steam,
    # 331k users) S can blow up to billions of nonzeros and 100+ GB even sparse.
    # Instead, expose S via a LinearOperator whose matvec computes
    #     (S - I) @ v  =  B_norm @ (B_normᵀ @ v) - v
    # on the fly. Two sparse matvecs per Lanczos step, ~100 MB total.
    # eigsh on the symmetric operator with which='LM' is the equivalent of the
    # original `svds(S - I, k)` since for a symmetric matrix singular values =
    # absolute eigenvalues.
    k = args.cf_emb_dim
    if k >= num_nonempty:
        raise SystemExit(f"--cf-emb-dim={k} must be < num_nonempty_users={num_nonempty}")
    BT = B_norm.T.tocsr()  # cache for fast matvecs
    n = B_norm.shape[0]

    def s_minus_i_matvec(v: np.ndarray) -> np.ndarray:
        return B_norm @ (BT @ v) - v

    S_op = LinearOperator(shape=(n, n), matvec=s_minus_i_matvec,
                          rmatvec=s_minus_i_matvec, dtype=np.float64)

    t0 = time.time()
    eigvals, U = eigsh(S_op, k=k, which="LM")  # top-k by |eigenvalue|
    order = np.argsort(-np.abs(eigvals))
    U = U[:, order]
    sigma = np.abs(eigvals[order])
    print(f"truncated eigen-decomp k={k} via LinearOperator: "
          f"sigma[0]={sigma[0]:.4f} sigma[-1]={sigma[-1]:.4f} in {time.time()-t0:.1f}s")

    sigma = np.clip(sigma, a_min=0.0, a_max=None)
    nonempty_emb = (U * np.sqrt(sigma)[None, :]).astype(np.float32)
    norms = np.linalg.norm(nonempty_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    nonempty_emb = nonempty_emb / norms

    # Expand to full table: empty users get a zero row.
    embeddings = np.zeros((num_users, k), dtype=np.float32)
    for nz_i, uid in enumerate(nonempty_uids):
        embeddings[user_idx[uid]] = nonempty_emb[nz_i]

    print(f"embeddings (full table): shape={embeddings.shape}  dtype={embeddings.dtype}")

    np.savez(
        args.output_path,
        user_ids=np.asarray(user_ids, dtype=object),
        embeddings=embeddings,
        has_cf=has_cf,
        dim=np.int32(k),
    )
    print(f"saved -> {args.output_path}")


if __name__ == "__main__":
    main()
