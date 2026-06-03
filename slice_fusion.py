#!/usr/bin/env python3
"""Per-cohort comparison of fusion(query, profile) vs fusion(query, 0) on test.

Isolates the profile-signal contribution within an existing fusion head by
running the same head twice — once with the cached profile embedding, once
with a zero profile — and computing the per-example rank delta. Designed to
surface whether profile signal helps a *cohort* even when aggregate metrics
are flat.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from data import (
    AMAZON2018_URLS,
    HistoryFormatConfig,
    build_eval_catalog,
    build_examples,
    build_user_sequences,
    load_interactions,
    load_item_texts,
)
from evaluation import encode_documents, encode_queries
from fusion import FusionHead, GatedProfileFusionHead
from profiles import load_profile_cache, profile_cache_key
from train_fusion import build_user_gate_features, fill_cos_q_p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqrec-checkpoint", type=str,
                   default="outputs/seqrec_qwen3_beauty/best")
    p.add_argument("--fusion-head-path", type=str,
                   default="outputs/fusion_beauty_v6cf/fusion_head.pt")
    p.add_argument("--llm-profile-cache", type=str,
                   default="outputs/profiles_beauty_v6_cot_cf.jsonl")
    p.add_argument("--llm-profile-max-history", type=int, default=20)
    p.add_argument("--dataset-name", type=str,
                   choices=sorted(AMAZON2018_URLS.keys()), default="Beauty")
    p.add_argument("--root", type=str, default="./raw_data")
    p.add_argument("--rating-score", type=float, default=0.0)
    p.add_argument("--download-if-missing", action="store_true", default=True)
    p.add_argument("--max-title-words", type=int, default=20)
    p.add_argument("--max-history-items", type=int, default=20)
    p.add_argument("--min-user-seq-len", type=int, default=3)
    p.add_argument("--enrich-text-input", action="store_true")
    p.add_argument("--eval-catalog", type=str,
                   choices=["interacted", "metadata"], default="interacted")
    p.add_argument("--encode-batch-size", type=int, default=8)
    p.add_argument("--scoring-chunk-size", type=int, default=2048)
    p.add_argument("--filter-seen-items", action="store_true",
                   help="Mask items in the user's history before ranking (matches train_fusion.py).")
    p.add_argument("--output-dir", type=str,
                   default="outputs/fusion_beauty_v6cf/slices")
    p.add_argument("--fusion-device", type=str, default="cuda")
    p.add_argument("--baseline-per-example", type=str,
                   default="outputs/seqrec_qwen3_beauty/slices/per_example.parquet",
                   help="Per-example parquet from the text-only baseline (evaluate_slices.py).")
    return p.parse_args()


@torch.no_grad()
def per_example_ranks_fused(
    head: torch.nn.Module,
    query_emb: torch.Tensor,
    profile_emb: torch.Tensor,
    catalog_emb: torch.Tensor,
    gold_indices: torch.Tensor,
    device: str,
    chunk_size: int,
    has_profile: torch.Tensor | None = None,
    hist_feats: torch.Tensor | None = None,
    bypass_head: bool = False,
    seen_indices: List[torch.Tensor] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ranks, gate_values). gate_values is all-NaN for non-gated heads.

    When `bypass_head=True`, the head is skipped and ranking falls back to the
    text-only baseline `normalize(q) @ catalog.T`. Used to ablate the profile
    contribution of a non-gated head without taking the head off-distribution
    (concat([q, 0]) -> Linear(2D,D) does NOT reproduce the baseline).
    """
    head.eval()
    is_gated = isinstance(head, GatedProfileFusionHead)
    n_q = query_emb.shape[0]
    catalog_dev = catalog_emb.to(device)
    ranks = np.zeros(n_q, dtype=np.int64)
    gate_vals = np.full(n_q, np.nan, dtype=np.float32)
    for s in range(0, n_q, chunk_size):
        e = min(s + chunk_size, n_q)
        q = query_emb[s:e].to(device)
        p = profile_emb[s:e].to(device)
        if bypass_head:
            fused = torch.nn.functional.normalize(q, dim=-1)
        elif is_gated:
            hp = has_profile[s:e].to(device)
            hf = hist_feats[s:e].to(device)
            fused = head(q, p, hp, hf)
            gate_vals[s:e] = head.gate(hf, hp).squeeze(-1).cpu().numpy()
        else:
            fused = head(q, p)
        scores = fused @ catalog_dev.T                # (b, C)
        if seen_indices is not None:
            for j, idx in enumerate(seen_indices[s:e]):
                if idx.numel() > 0:
                    scores[j, idx.to(device)] = float("-inf")
        gold = gold_indices[s:e].to(device)
        gold_scores = scores.gather(1, gold.view(-1, 1)).squeeze(1)
        higher = (scores > gold_scores.unsqueeze(1)).sum(dim=1)
        ranks[s:e] = (higher + 1).cpu().numpy()
    return ranks, gate_vals


def _ndcg10(rank: np.ndarray) -> float:
    r = rank.astype(np.float64)
    return float(np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0).mean())


def _focus_report(df: pd.DataFrame, hl_label: str, pop_label: str, has_bl: bool) -> None:
    print(f"\n=== Focus: history {hl_label} x target {pop_label} ===")
    focus = df[(df["hl"] == hl_label) & (df["pop"] == pop_label)]
    print(f"  n = {len(focus)}")
    if len(focus) == 0:
        return
    if has_bl:
        print(f"  R@10    baseline   : {focus['hit10_baseline'].mean():.4f}")
        print(f"  NDCG@10 baseline   : {_ndcg10(focus['rank_baseline'].values):.4f}")
    print(f"  R@10    fused+prof : {focus['hit10_prof'].mean():.4f}")
    print(f"  NDCG@10 fused+prof : {_ndcg10(focus['rank_prof'].values):.4f}")
    print(f"  R@10    fused+zero : {focus['hit10_zero'].mean():.4f}")
    print(f"  NDCG@10 fused+zero : {_ndcg10(focus['rank_zero'].values):.4f}")
    if has_bl:
        bl_hit = focus["hit10_baseline"].astype(bool).values
        pf_hit = focus["hit10_prof"].astype(bool).values
        gained = int((pf_hit & ~bl_hit).sum())
        lost = int((~pf_hit & bl_hit).sum())
        kept = int((pf_hit & bl_hit).sum())
        miss = int((~pf_hit & ~bl_hit).sum())
        print(f"  baseline→fused+prof transitions: "
              f"gained={gained} lost={lost} kept={kept} both_miss={miss}")


def main() -> None:
    args = parse_args()
    device = args.fusion_device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # --- match the training-time history format
    fmt = HistoryFormatConfig(
        time_text=args.enrich_text_input,
        rating_text=args.enrich_text_input,
        sep=args.enrich_text_input,
        pos_marker=args.enrich_text_input,
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
    _train, _val, test_examples = build_examples(
        user_sequences=user_sequences,
        item_to_text=item_to_text,
        min_user_seq_len=args.min_user_seq_len,
        max_history_items=args.max_history_items,
        fmt=fmt,
    )

    eval_item_to_text = build_eval_catalog(
        item_to_text=item_to_text,
        interactions=interactions,
        eval_catalog=args.eval_catalog,
    )
    item_ids = sorted(eval_item_to_text.keys())
    id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    catalog_texts = [eval_item_to_text[i] for i in item_ids]

    # --- profile cache: map every test example to (profile_string | None)
    profile_cache = load_profile_cache(args.llm_profile_cache)
    if not profile_cache:
        raise SystemExit(f"Profile cache at {args.llm_profile_cache!r} is empty.")
    print(f"Loaded {len(profile_cache):,} profile entries from cache.")

    profile_strs: List[str] = []
    has_profile: List[bool] = []
    for ex in test_examples:
        key = profile_cache_key(ex.history_item_ids, args.llm_profile_max_history)
        prof = profile_cache.get(key)
        has_profile.append(prof is not None)
        profile_strs.append(prof if prof is not None else "")
    n_missing = sum(1 for h in has_profile if not h)
    print(f"Test examples: {len(test_examples):,} total, {n_missing:,} without profile.")

    # --- encode
    print(f"Loading encoder from {args.seqrec_checkpoint}")
    encoder = SentenceTransformer(args.seqrec_checkpoint)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    embed_dim = encoder.get_sentence_embedding_dimension()

    print(f"Encoding catalog ({len(item_ids):,} items)...")
    catalog_np = encode_documents(encoder, catalog_texts,
                                  batch_size=args.encode_batch_size).astype(np.float32, copy=False)
    catalog_emb = torch.from_numpy(catalog_np)

    print(f"Encoding queries ({len(test_examples):,})...")
    queries = [ex.history_text for ex in test_examples]
    query_np = encode_queries(encoder, queries,
                              batch_size=args.encode_batch_size).astype(np.float32, copy=False)

    print(f"Encoding profile strings (deduped)...")
    # Dedup non-empty profiles
    uniq: Dict[str, int] = {}
    uniq_list: List[str] = []
    for s in profile_strs:
        if s and s not in uniq:
            uniq[s] = len(uniq_list)
            uniq_list.append(s)
    if uniq_list:
        uniq_emb = encode_queries(encoder, uniq_list,
                                  batch_size=args.encode_batch_size).astype(np.float32, copy=False)
    else:
        uniq_emb = np.zeros((0, embed_dim), dtype=np.float32)
    profile_np = np.zeros((len(test_examples), embed_dim), dtype=np.float32)
    for i, s in enumerate(profile_strs):
        if s:
            profile_np[i] = uniq_emb[uniq[s]]

    query_emb = torch.from_numpy(query_np)
    profile_emb = torch.from_numpy(profile_np)
    profile_zero = torch.zeros_like(profile_emb)

    # --- fusion head
    print(f"Loading fusion head from {args.fusion_head_path}")
    ckpt = torch.load(args.fusion_head_path, map_location="cpu")
    head_type = ckpt.get("head_type", "mlp")
    mlp_hidden = int(ckpt.get("mlp_hidden", 512))
    if head_type == "gated_profile":
        gate_features = ckpt["gate_features"]
        head: torch.nn.Module = GatedProfileFusionHead(
            embed_dim=embed_dim,
            num_gate_features=len(gate_features),
            gate_mlp_hidden=int(ckpt.get("gate_mlp_hidden", 16)),
            gate_logit_init=float(ckpt.get("gate_logit_init", -6.0)),
        )
        print(f"  gated_profile head; gate_features={gate_features}")
    else:
        head = FusionHead(embed_dim=embed_dim, head_type=head_type, mlp_hidden=mlp_hidden)
    head.load_state_dict(ckpt["state_dict"])
    head.to(device)

    # --- gate inputs (gated_profile only)
    has_profile_t = hist_feats_t = None
    has_profile_zero_t = None
    if isinstance(head, GatedProfileFusionHead):
        log_pop_path = ckpt.get("item_log_pop_path") or os.path.join(
            os.path.dirname(os.path.abspath(args.fusion_head_path)), "item_log_pop.json"
        )
        with open(log_pop_path) as f:
            item_log_pop = json.load(f)
        has_profile_arr = np.asarray(has_profile, dtype=np.float32)
        feats_np = build_user_gate_features(
            test_examples, has_profile_arr.tolist(), item_log_pop, gate_features,
        )
        fill_cos_q_p(feats_np, gate_features, query_np, profile_np, has_profile_arr)
        hist_feats_t = torch.from_numpy(feats_np).float()
        has_profile_t = torch.from_numpy(has_profile_arr).float().unsqueeze(-1)
        # For the zero-profile pass: gate features set has_profile=0 and cos_q_p=0,
        # so the gate self-zeros and the residual contribution is forced off.
        feats_zero_np = feats_np.copy()
        zero_has_profile = np.zeros_like(has_profile_arr)
        if "has_profile" in gate_features:
            feats_zero_np[:, gate_features.index("has_profile")] = 0.0
        if "cos_q_p" in gate_features:
            feats_zero_np[:, gate_features.index("cos_q_p")] = 0.0
        hist_feats_zero_t = torch.from_numpy(feats_zero_np).float()
        has_profile_zero_t = torch.from_numpy(zero_has_profile).float().unsqueeze(-1)
    else:
        hist_feats_zero_t = None

    gold_indices = torch.tensor(
        [id_to_idx[ex.target_item_id] for ex in test_examples], dtype=torch.long
    )

    seen_indices_t: List[torch.Tensor] | None = None
    if args.filter_seen_items:
        seen_indices_t = []
        for ex in test_examples:
            sidx = sorted({id_to_idx[i] for i in ex.history_item_ids if i in id_to_idx})
            seen_indices_t.append(torch.tensor(sidx, dtype=torch.long))
        mean_seen = float(np.mean([t.numel() for t in seen_indices_t]))
        print(f"filter_seen_items=True; mean seen-items per query: {mean_seen:.2f}")

    print("Computing ranks WITH profile...")
    rank_prof, gate_prof = per_example_ranks_fused(
        head, query_emb, profile_emb, catalog_emb, gold_indices,
        device=device, chunk_size=args.scoring_chunk_size,
        has_profile=has_profile_t, hist_feats=hist_feats_t,
        seen_indices=seen_indices_t,
    )
    is_gated = isinstance(head, GatedProfileFusionHead)
    if is_gated:
        print("Computing ranks WITH zero profile (gate forced off)...")
    else:
        print("Computing ranks WITH zero profile (head bypassed -> text-only baseline)...")
    rank_zero, _gate_zero = per_example_ranks_fused(
        head, query_emb, profile_zero, catalog_emb, gold_indices,
        device=device, chunk_size=args.scoring_chunk_size,
        has_profile=has_profile_zero_t, hist_feats=hist_feats_zero_t,
        bypass_head=not is_gated,
        seen_indices=seen_indices_t,
    )

    # --- per-example dataframe + cohort columns (match evaluate_slices.py)
    item_freq_total = Counter(x.item_id for x in interactions)
    history_len = np.array([len(ex.history_item_ids) for ex in test_examples], dtype=np.int64)
    target_freq = np.array(
        [max(0, item_freq_total[ex.target_item_id] - 1) for ex in test_examples], dtype=np.int64,
    )

    df = pd.DataFrame({
        "user_id": [ex.user_id for ex in test_examples],
        "target_item_id": [ex.target_item_id for ex in test_examples],
        "history_len": history_len,
        "target_freq": target_freq,
        "has_profile": np.array(has_profile, dtype=bool),
        "rank_prof": rank_prof,
        "rank_zero": rank_zero,
        "hit10_prof": rank_prof <= 10,
        "hit10_zero": rank_zero <= 10,
        "gate": gate_prof,
    })

    # join baseline (text-only) ranks if available
    if args.baseline_per_example and os.path.exists(args.baseline_per_example):
        bl = pd.read_parquet(args.baseline_per_example)[
            ["user_id", "target_item_id", "rank"]
        ].rename(columns={"rank": "rank_baseline"})
        df = df.merge(bl, on=["user_id", "target_item_id"], how="left")
        df["hit10_baseline"] = df["rank_baseline"] <= 10
        n_joined = df["rank_baseline"].notna().sum()
        print(f"Joined baseline ranks for {n_joined:,}/{len(df):,} examples.")
    else:
        df["rank_baseline"] = np.nan
        df["hit10_baseline"] = False

    out_parquet = os.path.join(args.output_dir, "per_example_prof_vs_zero.parquet")
    df.to_parquet(out_parquet, index=False)
    print(f"\nWrote per-example -> {out_parquet}")

    # --- aggregate + cohort tables
    def hl_bin(h: int) -> str:
        if h <= 5:  return "3-5"
        if h <= 10: return "6-10"
        if h <= 20: return "11-20"
        return "21+"

    pop_edges = [
        float(np.quantile(target_freq, q)) for q in (0.2, 0.4, 0.6, 0.8)
    ]

    def pop_bin(f: int) -> str:
        if f <= pop_edges[0]: return "Q1"
        if f <= pop_edges[1]: return "Q2"
        if f <= pop_edges[2]: return "Q3"
        if f <= pop_edges[3]: return "Q4"
        return "Q5"

    df["hl"] = df["history_len"].map(hl_bin)
    df["pop"] = df["target_freq"].map(pop_bin)

    has_bl = df["rank_baseline"].notna().any()

    print(f"\n=== Aggregate (n={len(df)}) ===")
    if has_bl:
        print(f"  R@10 baseline (text only)       : {df['hit10_baseline'].mean():.4f}")
    print(f"  R@10 fused, WITH profile        : {df['hit10_prof'].mean():.4f}")
    print(f"  R@10 fused, ZERO profile        : {df['hit10_zero'].mean():.4f}")
    if has_bl:
        print(f"  delta (fused+prof - baseline)    : "
              f"{df['hit10_prof'].mean() - df['hit10_baseline'].mean():+.4f}")
    print(f"  delta (fused+prof - fused+zero) : "
          f"{df['hit10_prof'].mean() - df['hit10_zero'].mean():+.4f}")

    hl_order = ["3-5", "6-10", "11-20", "21+"]
    pop_order = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def pivot(col: str) -> pd.DataFrame:
        return (df.pivot_table(index="hl", columns="pop", values=col, aggfunc="mean")
                  .reindex(hl_order)[pop_order] * 100).round(2)

    cnt = (df.pivot_table(index="hl", columns="pop", values="hit10_prof", aggfunc="size")
             .reindex(hl_order)[pop_order])

    if has_bl:
        print("\nR@10 BASELINE (text only):")
        print(pivot("hit10_baseline").to_string())
    print("\nR@10 FUSED + profile:")
    print(pivot("hit10_prof").to_string())
    print("\nR@10 FUSED + zero profile:")
    print(pivot("hit10_zero").to_string())
    if has_bl:
        print("\nDELTA (fused+prof - baseline), pp:")
        print((pivot("hit10_prof") - pivot("hit10_baseline")).to_string())
    print("\nDELTA (fused+prof - fused+zero), pp:  # profile signal within fusion")
    print((pivot("hit10_prof") - pivot("hit10_zero")).to_string())
    print("\nCounts:")
    print(cnt.to_string())

    if not df["gate"].isna().all():
        gate_pivot = (df.pivot_table(index="hl", columns="pop", values="gate", aggfunc="mean")
                        .reindex(hl_order)[pop_order]).round(4)
        print(f"\nMEAN GATE per cohort (global mean = {df['gate'].mean():.4f}):")
        print(gate_pivot.to_string())

    _focus_report(df, "3-5", "Q5", has_bl)
    _focus_report(df, "21+", "Q1", has_bl)


if __name__ == "__main__":
    main()
