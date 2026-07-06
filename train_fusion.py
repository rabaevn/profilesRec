#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from config import parse_fusion_args
from data import (
    Example,
    HistoryFormatConfig,
    build_eval_catalog,
    build_examples,
    build_user_sequences,
    load_interactions,
    load_item_texts,
)
from evaluation import encode_documents, encode_queries
from fusion import (
    FusionHead,
    FusionTripleDataset,
    GatedFusionDataset,
    GatedProfileFusionHead,
    info_nce_loss,
)
from profiles import load_profile_cache, profile_cache_key
from utils import save_json, set_seed


GATE_FEATURE_NAMES = ("history_len", "has_profile", "mean_history_pop", "cos_q_p")


def parse_gate_features(spec: str) -> List[str]:
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in GATE_FEATURE_NAMES]
    if bad:
        raise ValueError(
            f"Unknown gate feature(s) {bad}. Valid: {list(GATE_FEATURE_NAMES)}"
        )
    if not names:
        raise ValueError("--gate-features must include at least one feature.")
    return names


def compute_item_log_pop(interactions) -> Dict[str, float]:
    """log1p of corpus-wide item frequency. Used only for history items (not targets)."""
    counts = Counter(x.item_id for x in interactions)
    return {iid: float(math.log1p(c)) for iid, c in counts.items()}


def build_sample_weights(examples: Sequence[Example], scheme: str) -> np.ndarray:
    """Per-example InfoNCE weight derived from len(history_item_ids).

    Schemes:
      - none:      uniform (all 1.0)
      - log_hl:    log1p(history_len)         [hl=3->1.39, hl=20->3.04]
      - sqrt_hl:   sqrt(history_len)          [hl=3->1.73, hl=20->4.47]
      - linear_hl: history_len                [hl=3->3.00, hl=20->20.0]
    """
    hl = np.asarray([len(ex.history_item_ids) for ex in examples], dtype=np.float64)
    if scheme == "none":
        w = np.ones_like(hl)
    elif scheme == "log_hl":
        w = np.log1p(hl)
    elif scheme == "sqrt_hl":
        w = np.sqrt(hl)
    elif scheme == "linear_hl":
        w = hl.copy()
    else:
        raise ValueError(f"Unknown sample-reweight scheme: {scheme!r}")
    # Guard against degenerate (all-zero) weights for hl=0 edge cases.
    w = np.maximum(w, 1e-3)
    return w.astype(np.float32)


def build_user_gate_features(
    examples: Sequence[Example],
    has_profile: Sequence[float],
    item_log_pop: Dict[str, float],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Build the user-only columns of gate features (cos_q_p is filled in later)."""
    rows: List[List[float]] = []
    for ex, hp in zip(examples, has_profile):
        hist_len = len(ex.history_item_ids)
        if hist_len > 0:
            mean_pop = float(
                np.mean([item_log_pop.get(i, 0.0) for i in ex.history_item_ids])
            )
        else:
            mean_pop = 0.0
        row: List[float] = []
        for name in feature_names:
            if name == "history_len":
                row.append(math.log1p(hist_len))
            elif name == "has_profile":
                row.append(float(hp))
            elif name == "mean_history_pop":
                row.append(mean_pop)
            elif name == "cos_q_p":
                row.append(0.0)  # filled in after encoding
            else:
                raise ValueError(f"Unknown gate feature: {name}")
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def fill_cos_q_p(
    feats: np.ndarray,
    feature_names: Sequence[str],
    q_emb: np.ndarray,
    p_emb: np.ndarray,
    has_profile: np.ndarray,
) -> None:
    """In-place: write cos(q, p) * has_profile into the cos_q_p column, if present."""
    if "cos_q_p" not in feature_names:
        return
    col = feature_names.index("cos_q_p")
    # q_emb and p_emb come from the SentenceTransformer pipeline which L2-normalizes;
    # be defensive and renormalize so cos is always well-defined.
    qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    pn = p_emb / (np.linalg.norm(p_emb, axis=1, keepdims=True) + 1e-12)
    cos = (qn * pn).sum(axis=1) * has_profile
    feats[:, col] = cos.astype(np.float32, copy=False)


@torch.no_grad()
def compute_oracle_labels(
    query_np: np.ndarray,
    profile_np: np.ndarray,
    catalog_np: np.ndarray,
    gold_indices: np.ndarray,
    has_profile: np.ndarray,
    alpha: float,
    device: str,
    chunk_size: int,
    objective: str = "any_uplift",
) -> np.ndarray:
    """Per-example oracle label.

    For each example i:
        s_text  = q_i · catalog
        s_fused = (1-α)·(q_i · catalog) + α·(p_i · catalog)
        rank_text  = #{c : s_text(c) > s_text(gold)}    # 0-indexed
        rank_fused = #{c : s_fused(c) > s_fused(gold)}

    objective="any_uplift":     y_i = 1[ rank_fused < rank_text ]
        Trains the gate for soft re-ranking — any upward movement of the gold.
    objective="r10_recovery":   y_i = 1[ rank_text >= 10 AND rank_fused < 10 ]
        Trains the gate for Recall@10 boundary crossings — fires only when
        text-only misses top-10 and the profile mix recovers the gold into top-10.
        Note: rank here is 0-indexed (#strictly-higher), so rank<10 means the gold
        is among the top 10. Sparse positive class — pair with --gate-aux-pos-weight.

    has_profile=0 rows are forced to 0 (no signal to learn from).
    """
    if objective not in ("any_uplift", "r10_recovery"):
        raise ValueError(f"Unknown gate_aux_objective={objective!r}")
    n = query_np.shape[0]
    catalog_t = torch.from_numpy(catalog_np).to(device).float()
    gold_t = torch.from_numpy(gold_indices).to(device).long()
    hp_t = torch.from_numpy(has_profile.astype(np.float32)).to(device)
    y = np.zeros(n, dtype=np.float32)
    a = float(alpha)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        q = torch.from_numpy(query_np[s:e]).to(device).float()
        p = torch.from_numpy(profile_np[s:e]).to(device).float()
        s_text = q @ catalog_t.T  # (b, C)
        s_fused = (1.0 - a) * s_text + a * (p @ catalog_t.T)
        gold_chunk = gold_t[s:e].view(-1, 1)
        gold_text = s_text.gather(1, gold_chunk).squeeze(1)
        gold_fused = s_fused.gather(1, gold_chunk).squeeze(1)
        rank_text = (s_text > gold_text.unsqueeze(1)).sum(dim=1)
        rank_fused = (s_fused > gold_fused.unsqueeze(1)).sum(dim=1)
        if objective == "any_uplift":
            y_chunk = (rank_fused < rank_text).float()
        else:  # r10_recovery
            y_chunk = ((rank_text >= 10) & (rank_fused < 10)).float()
        y_chunk = y_chunk * hp_t[s:e]
        y[s:e] = y_chunk.cpu().numpy()
    return y


def resolve_profile_strings(
    examples: Sequence[Example],
    profile_cache: Dict[str, str],
    max_history: int,
    keep_missing: bool = False,
) -> Tuple[List[Example], List[str], List[float]]:
    """Resolve profile strings for each example.

    When `keep_missing` is False (legacy behavior, used by the non-gated heads),
    examples without a profile cache entry are dropped.

    When `keep_missing` is True (used by gated_profile), all examples are kept;
    missing entries get an empty profile string and `has_profile = 0.0` so the
    gate can mask them out at training/inference time.

    Returns (examples, profile_strings, has_profile) — all the same length.
    """
    kept: List[Example] = []
    profiles: List[str] = []
    has_profile: List[float] = []
    missing = 0
    for ex in examples:
        key = profile_cache_key(ex.history_item_ids, max_history)
        prof = profile_cache.get(key)
        if prof is None:
            missing += 1
            if not keep_missing:
                continue
            kept.append(ex)
            profiles.append("")
            has_profile.append(0.0)
        else:
            kept.append(ex)
            profiles.append(prof)
            has_profile.append(1.0)
    if missing:
        msg = f"  profile cache: {missing:,} examples without entry"
        msg += " (kept with has_profile=0)" if keep_missing else " (dropped)"
        print(msg)
    return kept, profiles, has_profile


def _texts_fingerprint(texts: Sequence[str], salt: str = "") -> str:
    """Order-sensitive content hash — cache is valid only for the exact text list
    (and salt, which should identify the encoder so a retrained backbone at the
    same path does not reuse stale embeddings)."""
    h = hashlib.sha1()
    h.update(f"{salt}\x1e{len(texts)}\x1e".encode())
    for t in texts:
        h.update(t.encode("utf-8", "surrogatepass"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def encoder_cache_salt(checkpoint_path: str) -> str:
    """Identify the encoder by checkpoint path + mtime (changes on retrain)."""
    try:
        mtime = int(os.path.getmtime(checkpoint_path))
    except OSError:
        mtime = 0
    return f"{checkpoint_path}:{mtime}"


def _atomic_np_save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def encode_texts_cached(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
    is_query: bool,
    cache_dir: str,
    cache_name: str,
    cache_salt: str = "",
    shard_size: int = 100_000,
) -> np.ndarray:
    """Encode texts, checkpointing shards to disk so an interrupted run resumes.

    Cache files are keyed by a content fingerprint of `texts` (+ encoder salt),
    so a stale cache (different encoder inputs) is simply ignored rather than
    reused.
    """
    encode_fn = encode_queries if is_query else encode_documents
    if not cache_dir:
        return encode_fn(model, list(texts), batch_size=batch_size).astype(
            np.float32, copy=False
        )

    os.makedirs(cache_dir, exist_ok=True)
    fp = _texts_fingerprint(texts, salt=cache_salt)
    final_path = os.path.join(cache_dir, f"{cache_name}_{fp}.npy")
    if os.path.exists(final_path):
        print(f"  emb cache hit: {final_path}", flush=True)
        return np.load(final_path)

    n = len(texts)
    dim = model.get_sentence_embedding_dimension()
    out = np.empty((n, dim), dtype=np.float32)
    num_shards = (n + shard_size - 1) // shard_size
    for shard_i, s in enumerate(range(0, n, shard_size)):
        e = min(s + shard_size, n)
        shard_path = os.path.join(cache_dir, f"{cache_name}_{fp}_shard{shard_i:05d}.npy")
        if os.path.exists(shard_path):
            emb = np.load(shard_path)
            if emb.shape != (e - s, dim):
                raise RuntimeError(
                    f"Corrupt emb cache shard {shard_path}: "
                    f"shape={emb.shape}, expected {(e - s, dim)}"
                )
            print(f"  emb cache shard {shard_i + 1}/{num_shards} loaded", flush=True)
        else:
            emb = encode_fn(model, list(texts[s:e]), batch_size=batch_size).astype(
                np.float32, copy=False
            )
            _atomic_np_save(shard_path, emb)
            print(f"  emb cache shard {shard_i + 1}/{num_shards} encoded+saved", flush=True)
        out[s:e] = emb
    _atomic_np_save(final_path, out)
    for shard_i in range(num_shards):
        shard_path = os.path.join(cache_dir, f"{cache_name}_{fp}_shard{shard_i:05d}.npy")
        if os.path.exists(shard_path):
            os.remove(shard_path)
    print(f"  emb cache saved: {final_path}", flush=True)
    return out


def encode_with_dedup(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
    is_query: bool,
    cache_dir: str = "",
    cache_name: str = "",
    cache_salt: str = "",
) -> np.ndarray:
    """Encode unique texts once, then index back to the full list."""
    unique_to_idx: Dict[str, int] = {}
    ordered_unique: List[str] = []
    indices: List[int] = []
    for t in texts:
        if t not in unique_to_idx:
            unique_to_idx[t] = len(ordered_unique)
            ordered_unique.append(t)
        indices.append(unique_to_idx[t])

    unique_emb = encode_texts_cached(
        model, ordered_unique, batch_size, is_query,
        cache_dir=cache_dir, cache_name=cache_name, cache_salt=cache_salt,
    )
    return unique_emb[np.asarray(indices, dtype=np.int64)]


def encode_split(
    model: SentenceTransformer,
    examples: Sequence[Example],
    profile_strings: Sequence[str],
    item_id_to_index: Dict[str, int],
    encode_batch_size: int,
    has_profile: Sequence[float] | None = None,
    cache_dir: str = "",
    split_name: str = "",
    cache_salt: str = "",
    free_history_text: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Encode a split. Returns query/profile/target-index tensors plus the numpy
    query_emb, numpy profile_emb (zeroed where has_profile==0), and has_profile
    array — so callers can build gate features (e.g. cos(q, p)).

    Targets are returned as catalog *indices* (not embeddings): materializing
    catalog rows per example costs n_examples x embed_dim float32 (~12G on
    Steam) for rows already present in catalog_emb.

    free_history_text drops each example's history_text once queries are
    encoded — on Steam that's tens of GB of strings not needed afterwards.
    """
    queries = [x.history_text for x in examples]
    query_emb = encode_with_dedup(
        model, queries, encode_batch_size, is_query=True,
        cache_dir=cache_dir, cache_name=f"{split_name}_query", cache_salt=cache_salt,
    )
    del queries
    if free_history_text:
        for ex in examples:
            object.__setattr__(ex, "history_text", "")  # Example is frozen
    profile_emb = encode_with_dedup(
        model, list(profile_strings), encode_batch_size, is_query=True,
        cache_dir=cache_dir, cache_name=f"{split_name}_profile", cache_salt=cache_salt,
    )

    if has_profile is None:
        has_profile_arr = np.ones(len(examples), dtype=np.float32)
    else:
        has_profile_arr = np.asarray(has_profile, dtype=np.float32)
        # Zero out any rows that came from an empty profile string.
        profile_emb = profile_emb * has_profile_arr[:, None]

    target_indices = np.asarray(
        [item_id_to_index[x.target_item_id] for x in examples], dtype=np.int64
    )

    return (
        torch.from_numpy(query_emb).float(),
        torch.from_numpy(profile_emb).float(),
        torch.from_numpy(target_indices),
        query_emb,
        profile_emb,
        has_profile_arr,
    )


@torch.no_grad()
def fused_topk(
    head: torch.nn.Module,
    query_emb: torch.Tensor,
    profile_emb: torch.Tensor,
    catalog_emb: torch.Tensor,
    seen_indices: List[torch.Tensor],
    max_k: int,
    device: str,
    chunk_size: int = 1024,
    has_profile: torch.Tensor | None = None,
    hist_feats: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the fusion head and return top-K (scores, indices) per query."""
    head.eval()
    is_gated = isinstance(head, GatedProfileFusionHead)
    if is_gated and (has_profile is None or hist_feats is None):
        raise ValueError("GatedProfileFusionHead requires has_profile and hist_feats.")
    num_q = query_emb.shape[0]
    catalog_dev = catalog_emb.to(device)
    top_scores_chunks: List[torch.Tensor] = []
    top_idx_chunks: List[torch.Tensor] = []
    for s in range(0, num_q, chunk_size):
        e = min(s + chunk_size, num_q)
        q = query_emb[s:e].to(device)
        p = profile_emb[s:e].to(device)
        if is_gated:
            hp = has_profile[s:e].to(device)
            hf = hist_feats[s:e].to(device)
            fused = head(q, p, hp, hf)
        else:
            fused = head(q, p)                              # (chunk, D) on device
        scores = fused @ catalog_dev.T                      # (chunk, C) on device
        if seen_indices:
            for j, idx in enumerate(seen_indices[s:e]):
                if idx.numel() > 0:
                    scores[j, idx.to(device)] = float("-inf")
        ts, ti = torch.topk(scores, k=max_k, dim=1)
        top_scores_chunks.append(ts.cpu())
        top_idx_chunks.append(ti.cpu())
    top_scores = torch.cat(top_scores_chunks, dim=0)
    top_idx = torch.cat(top_idx_chunks, dim=0)
    return top_scores, top_idx


def metrics_from_topk(
    top_idx: torch.Tensor,
    gold_indices: torch.Tensor,
    ks: Sequence[int],
) -> Dict[str, float]:
    """Recall@K / NDCG@K given a pre-sorted (N, max_k) top_idx tensor."""
    unique_ks = sorted({int(k) for k in ks if int(k) > 0})
    max_k = top_idx.shape[1]
    num_q = top_idx.shape[0]

    gold = gold_indices.view(-1, 1)
    hit_mask = (top_idx == gold)  # (N, max_k)
    hit_ranks = torch.where(
        hit_mask.any(dim=1),
        hit_mask.float().argmax(dim=1) + 1,
        torch.full((num_q,), max_k + 1, dtype=torch.long),
    )

    metrics: Dict[str, float] = {}
    for k in unique_ks:
        eff_k = min(k, max_k)
        hits = (hit_ranks <= eff_k).float()
        ndcg = torch.where(
            hits.bool(),
            1.0 / torch.log2(hit_ranks.float() + 1.0),
            torch.zeros_like(hits),
        )
        metrics[f"recall@{k}"] = float(hits.mean().item())
        metrics[f"ndcg@{k}"] = float(ndcg.mean().item())
    return metrics


@torch.no_grad()
def eval_full_ranking(
    head: torch.nn.Module,
    query_emb: torch.Tensor,
    profile_emb: torch.Tensor,
    catalog_emb: torch.Tensor,
    gold_indices: torch.Tensor,
    seen_indices: List[torch.Tensor],
    ks: Sequence[int],
    device: str,
    chunk_size: int = 1024,
    has_profile: torch.Tensor | None = None,
    hist_feats: torch.Tensor | None = None,
) -> Dict[str, float]:
    unique_ks = sorted({int(k) for k in ks if int(k) > 0})
    max_k = min(max(unique_ks), catalog_emb.shape[0])
    _scores, top_idx = fused_topk(
        head=head,
        query_emb=query_emb,
        profile_emb=profile_emb,
        catalog_emb=catalog_emb,
        seen_indices=seen_indices,
        max_k=max_k,
        device=device,
        chunk_size=chunk_size,
        has_profile=has_profile,
        hist_feats=hist_feats,
    )
    return metrics_from_topk(top_idx=top_idx, gold_indices=gold_indices, ks=ks)


def main() -> None:
    args = parse_fusion_args()
    set_seed(args.seed)
    os.makedirs(args.fusion_output_dir, exist_ok=True)
    device = args.fusion_device if torch.cuda.is_available() else "cpu"

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

    if args.enrich_text_input:
        args.history_sep = True
        args.history_time_text = True
        args.history_rating_text = True
        args.history_pos_marker = True
    history_fmt = HistoryFormatConfig(
        time_text=args.history_time_text,
        rating_text=args.history_rating_text,
        sep=args.history_sep,
        pos_marker=args.history_pos_marker,
    )
    train_examples, val_examples, _test_examples = build_examples(
        user_sequences=user_sequences,
        item_to_text=item_to_text,
        min_user_seq_len=args.min_user_seq_len,
        max_history_items=args.max_history_items,
        fmt=history_fmt,
    )

    eval_item_to_text = build_eval_catalog(
        item_to_text=item_to_text,
        interactions=interactions,
        eval_catalog=args.eval_catalog,
    )
    item_ids = sorted(eval_item_to_text.keys())
    item_id_to_index = {iid: i for i, iid in enumerate(item_ids)}
    catalog_texts = [eval_item_to_text[i] for i in item_ids]

    # ---- profiles
    # Only keys reachable from the train/val examples are ever looked up, so we
    # filter the cache at load time. On large datasets (Steam ~3M prefixes) this
    # is the difference between a few hundred MB and several GB of resident dict.
    needed_keys = {
        profile_cache_key(ex.history_item_ids, args.llm_profile_max_history)
        for ex in (*train_examples, *val_examples)
    }
    profile_cache = load_profile_cache(args.llm_profile_cache, keep_keys=needed_keys)
    if not profile_cache:
        raise SystemExit(
            f"Profile cache at {args.llm_profile_cache!r} is empty — run generate_profiles.py first."
        )
    print(
        f"Loaded {len(profile_cache):,} profiles from cache "
        f"({len(needed_keys):,} keys needed by examples)."
    )

    is_gated = args.fusion_head_type == "gated_profile"
    train_examples, train_profiles, train_has_prof = resolve_profile_strings(
        train_examples, profile_cache, args.llm_profile_max_history, keep_missing=is_gated
    )
    val_examples, val_profiles, val_has_prof = resolve_profile_strings(
        val_examples, profile_cache, args.llm_profile_max_history, keep_missing=is_gated
    )
    if not train_examples or not val_examples:
        raise SystemExit("No examples left after profile resolution — check the cache.")
    # The filtered cache is still millions of strings on large datasets; the
    # resolved per-split profile lists are all that's needed from here on.
    del profile_cache, needed_keys

    # ---- gate feature setup (gated_profile only)
    gate_feature_names: List[str] = []
    item_log_pop: Dict[str, float] = {}
    if is_gated:
        gate_feature_names = parse_gate_features(args.gate_features)
        print(f"Gate features: {gate_feature_names}")
        item_log_pop = compute_item_log_pop(interactions)
        log_pop_path = os.path.join(args.fusion_output_dir, "item_log_pop.json")
        with open(log_pop_path, "w") as f:
            json.dump(item_log_pop, f)
        print(f"Saved item log-popularity map -> {log_pop_path}")

    # Raw interactions/sequences are no longer needed once examples, the eval
    # catalog and item_log_pop exist (Steam: ~3.8M interaction objects).
    del interactions, user_sequences

    # ---- frozen encoder + one-shot embedding
    print(f"Loading frozen seqrec encoder from {args.seqrec_checkpoint}")
    encoder = SentenceTransformer(args.seqrec_checkpoint)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    embed_dim = encoder.get_sentence_embedding_dimension()
    print(f"seqrec embed_dim = {embed_dim}")

    emb_cache_dir = args.emb_cache_dir
    emb_cache_salt = encoder_cache_salt(args.seqrec_checkpoint) if emb_cache_dir else ""
    if emb_cache_dir:
        print(f"Embedding cache dir: {emb_cache_dir} (salt={emb_cache_salt})")

    print(f"Encoding catalog ({len(catalog_texts):,} items)...")
    catalog_emb_np = encode_texts_cached(
        encoder, catalog_texts, args.encode_batch_size, is_query=False,
        cache_dir=emb_cache_dir, cache_name="catalog", cache_salt=emb_cache_salt,
    )
    catalog_emb = torch.from_numpy(catalog_emb_np)

    print("Encoding train split...")
    train_q, train_p, train_t_idx, train_q_np, train_p_np, train_has_prof_np = encode_split(
        encoder, train_examples, train_profiles, item_id_to_index,
        args.encode_batch_size, has_profile=train_has_prof,
        cache_dir=emb_cache_dir, split_name="train", cache_salt=emb_cache_salt,
        free_history_text=True,
    )
    del train_profiles
    print("Encoding val split...")
    val_q, val_p, _val_t_idx, val_q_np, val_p_np, val_has_prof_np = encode_split(
        encoder, val_examples, val_profiles, item_id_to_index,
        args.encode_batch_size, has_profile=val_has_prof,
        cache_dir=emb_cache_dir, split_name="val", cache_salt=emb_cache_salt,
        free_history_text=True,
    )
    del val_profiles

    train_feats_t: torch.Tensor | None = None
    val_feats_t: torch.Tensor | None = None
    train_has_prof_t: torch.Tensor | None = None
    val_has_prof_t: torch.Tensor | None = None
    if is_gated:
        train_feats_np = build_user_gate_features(
            train_examples, train_has_prof_np.tolist(), item_log_pop, gate_feature_names,
        )
        val_feats_np = build_user_gate_features(
            val_examples, val_has_prof_np.tolist(), item_log_pop, gate_feature_names,
        )
        fill_cos_q_p(train_feats_np, gate_feature_names, train_q_np, train_p_np, train_has_prof_np)
        fill_cos_q_p(val_feats_np, gate_feature_names, val_q_np, val_p_np, val_has_prof_np)
        train_feats_t = torch.from_numpy(train_feats_np).float()
        val_feats_t = torch.from_numpy(val_feats_np).float()
        train_has_prof_t = torch.from_numpy(train_has_prof_np).float().unsqueeze(-1)
        val_has_prof_t = torch.from_numpy(val_has_prof_np).float().unsqueeze(-1)
        print(
            f"  train gate features: shape={tuple(train_feats_t.shape)} "
            f"has_profile mean train={float(train_has_prof_np.mean()):.3f} "
            f"val={float(val_has_prof_np.mean()):.3f}"
        )

    # ---- oracle labels for aux BCE loss (gated_profile + lambda > 0 only)
    train_y_oracle_t: torch.Tensor | None = None
    train_y_oracle_np: np.ndarray | None = None
    aux_pos_weight_t: torch.Tensor | None = None
    if is_gated and args.gate_aux_lambda > 0.0:
        train_gold_np = train_t_idx.numpy()
        print(
            f"Computing oracle labels: objective={args.gate_aux_objective} "
            f"alpha={args.gate_aux_alpha:.2f} "
            f"chunk_size={args.gate_aux_chunk_size} ..."
        )
        train_y_oracle_np = compute_oracle_labels(
            query_np=train_q_np,
            profile_np=train_p_np,
            catalog_np=catalog_emb_np,
            gold_indices=train_gold_np,
            has_profile=train_has_prof_np,
            alpha=args.gate_aux_alpha,
            device=device,
            chunk_size=args.gate_aux_chunk_size,
            objective=args.gate_aux_objective,
        )
        y_overall = float(train_y_oracle_np.mean())
        y_on_prof = float(
            train_y_oracle_np[train_has_prof_np > 0].mean()
            if (train_has_prof_np > 0).any() else 0.0
        )
        print(
            f"  oracle: y=1 fraction overall={y_overall:.4f} "
            f"on examples w/ profile={y_on_prof:.4f}"
        )
        hl_arr = np.asarray([len(ex.history_item_ids) for ex in train_examples])
        for lo, hi, label in [(3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10**9, "21+")]:
            m = (hl_arr >= lo) & (hl_arr <= hi)
            if m.any():
                print(
                    f"  hl={label:>5}: n={int(m.sum()):>6}  "
                    f"y=1 frac={float(train_y_oracle_np[m].mean()):.4f}"
                )
        oracle_path = os.path.join(args.fusion_output_dir, "oracle_labels.npy")
        np.save(oracle_path, train_y_oracle_np)
        print(f"  saved oracle labels -> {oracle_path}")
        train_y_oracle_t = torch.from_numpy(train_y_oracle_np).float()

        # BCE pos_weight for sparse positive classes (esp. r10_recovery).
        if args.gate_aux_pos_weight == "balanced":
            warm = train_has_prof_np > 0
            n_pos = float(((train_y_oracle_np > 0.5) & warm).sum())
            n_neg = float(((train_y_oracle_np <= 0.5) & warm).sum())
            if n_pos > 0:
                pw = n_neg / n_pos
                aux_pos_weight_t = torch.tensor(pw, dtype=torch.float32, device=device)
                print(f"  BCE pos_weight (balanced): n_pos={int(n_pos)} "
                      f"n_neg={int(n_neg)} pos_weight={pw:.3f}")
            else:
                print("  BCE pos_weight: requested 'balanced' but n_pos=0 — disabled")

    # ---- sample reweighting (gated path only)
    train_weights_t: torch.Tensor | None = None
    if is_gated and args.sample_reweight != "none":
        train_weights_np = build_sample_weights(train_examples, args.sample_reweight)
        hl_arr = np.asarray([len(ex.history_item_ids) for ex in train_examples])
        print(
            f"Sample reweight scheme={args.sample_reweight!r} "
            f"weight stats: mean={train_weights_np.mean():.3f} "
            f"min={train_weights_np.min():.3f} max={train_weights_np.max():.3f}"
        )
        for lo, hi, label in [(3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 10**9, "21+")]:
            mask = (hl_arr >= lo) & (hl_arr <= hi)
            if mask.any():
                print(
                    f"  hl={label:>5}: n={int(mask.sum()):>6}  "
                    f"mean_w={train_weights_np[mask].mean():.3f}"
                )
        train_weights_t = torch.from_numpy(train_weights_np).float()

    val_gold = torch.tensor(
        [item_id_to_index[x.target_item_id] for x in val_examples], dtype=torch.long
    )
    val_seen: List[torch.Tensor] = []
    if args.filter_seen_items:
        for ex in val_examples:
            seen_idx = sorted(
                {item_id_to_index[i] for i in ex.history_item_ids if i in item_id_to_index}
            )
            val_seen.append(torch.tensor(seen_idx, dtype=torch.long))

    # ---- head
    if is_gated:
        head = GatedProfileFusionHead(
            embed_dim=embed_dim,
            num_gate_features=len(gate_feature_names),
            gate_mlp_hidden=args.gate_mlp_hidden,
            gate_logit_init=args.gate_logit_init,
        ).to(device)
    else:
        head = FusionHead(
            embed_dim=embed_dim,
            head_type=args.fusion_head_type,
            mlp_hidden=args.fusion_mlp_hidden,
        ).to(device)

    if is_gated:
        gate_params = list(head.gate_mlp.parameters()) + list(head.feature_norm.parameters())
        gate_param_ids = {id(p) for p in gate_params}
        other_params = [p for p in head.parameters() if id(p) not in gate_param_ids]
        optim = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": args.fusion_weight_decay},
                {"params": gate_params, "weight_decay": args.gate_weight_decay},
            ],
            lr=args.fusion_learning_rate,
        )
    else:
        optim = torch.optim.AdamW(
            head.parameters(),
            lr=args.fusion_learning_rate,
            weight_decay=args.fusion_weight_decay,
        )

    # Targets are stored as catalog indices; the training loop gathers the
    # embedding rows from catalog_emb_device per batch.
    if is_gated:
        train_ds = GatedFusionDataset(
            train_q, train_p, train_t_idx, train_feats_t, train_has_prof_t,
            weights=train_weights_t,
            y_oracle=train_y_oracle_t,
        )
    else:
        train_ds = FusionTripleDataset(train_q, train_p, train_t_idx)
    loader = DataLoader(
        train_ds,
        batch_size=args.fusion_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    catalog_emb_device = catalog_emb.to(device)

    # ---- epoch-0 sanity (gated only): pre-train eval must equal text-only baseline
    if is_gated:
        sanity_metrics = eval_full_ranking(
            head=head,
            query_emb=val_q,
            profile_emb=val_p,
            catalog_emb=catalog_emb_device,
            gold_indices=val_gold,
            seen_indices=val_seen,
            ks=[10],
            device=device,
            has_profile=val_has_prof_t,
            hist_feats=val_feats_t,
        )
        print(
            f"[Epoch 0 sanity] val_recall@10={sanity_metrics['recall@10']:.4f} "
            f"(should equal text-only baseline; proj is zero-initialized)"
        )

    best_score = -math.inf
    best_metrics: Dict[str, float] = {}
    step = 0
    aux_enabled = is_gated and args.gate_aux_lambda > 0.0
    for epoch in range(1, args.fusion_num_epochs + 1):
        head.train()
        epoch_loss = 0.0
        epoch_loss_unweighted = 0.0
        epoch_n = 0
        epoch_gate_sum = 0.0
        epoch_gate_n = 0
        epoch_aux_loss_sum = 0.0
        epoch_aux_n = 0
        epoch_gate_y1_sum = 0.0
        epoch_gate_y1_n = 0
        epoch_gate_y0_sum = 0.0
        epoch_gate_y0_n = 0
        for batch in loader:
            if is_gated:
                q, p, t_idx, feats, hp, w, y_oracle = batch
                q = q.to(device)
                p = p.to(device)
                t = catalog_emb_device[t_idx.to(device)]
                feats = feats.to(device)
                hp = hp.to(device)
                w = w.to(device)
                y_oracle = y_oracle.to(device)
                fused = head(q, p, hp, feats)
                use_w = w if args.sample_reweight != "none" else None
                loss = info_nce_loss(
                    fused, t, temperature=args.fusion_temperature, weights=use_w
                )
                if args.gate_anchor_lambda > 0.0:
                    gate_vals = head.gate(feats, hp)
                    loss = loss + args.gate_anchor_lambda * gate_vals.mean()
                if aux_enabled:
                    # BCE on gate logit; restrict to has_profile==1 rows (the gate
                    # is masked to 0 anyway on the rest, so no signal to learn).
                    aux_mask = (hp.squeeze(-1) > 0.5)
                    if aux_mask.any():
                        aux_logit = head.gate_logit(feats).squeeze(-1)
                        aux_loss_per = torch.nn.functional.binary_cross_entropy_with_logits(
                            aux_logit[aux_mask], y_oracle[aux_mask], reduction="mean",
                            pos_weight=aux_pos_weight_t,
                        )
                        loss = loss + args.gate_aux_lambda * aux_loss_per
                        epoch_aux_loss_sum += float(aux_loss_per.item()) * int(aux_mask.sum().item())
                        epoch_aux_n += int(aux_mask.sum().item())
                with torch.no_grad():
                    gate_now = head.gate(feats, hp).squeeze(-1)
                    epoch_gate_sum += float(gate_now.mean().item()) * q.shape[0]
                    epoch_gate_n += q.shape[0]
                    if aux_enabled:
                        m1 = (y_oracle > 0.5) & (hp.squeeze(-1) > 0.5)
                        m0 = (y_oracle <= 0.5) & (hp.squeeze(-1) > 0.5)
                        if m1.any():
                            epoch_gate_y1_sum += float(gate_now[m1].sum().item())
                            epoch_gate_y1_n += int(m1.sum().item())
                        if m0.any():
                            epoch_gate_y0_sum += float(gate_now[m0].sum().item())
                            epoch_gate_y0_n += int(m0.sum().item())
                    if use_w is not None:
                        unw = info_nce_loss(fused, t, temperature=args.fusion_temperature)
                        epoch_loss_unweighted += float(unw.item()) * q.shape[0]
            else:
                q, p, t_idx = batch
                q = q.to(device)
                p = p.to(device)
                t = catalog_emb_device[t_idx.to(device)]
                fused = head(q, p)
                loss = info_nce_loss(fused, t, temperature=args.fusion_temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1
            epoch_loss += float(loss.item()) * q.shape[0]
            epoch_n += q.shape[0]

        avg_loss = epoch_loss / max(epoch_n, 1)
        avg_loss_unweighted = epoch_loss_unweighted / max(epoch_n, 1)
        val_metrics = eval_full_ranking(
            head=head,
            query_emb=val_q,
            profile_emb=val_p,
            catalog_emb=catalog_emb_device,
            gold_indices=val_gold,
            seen_indices=val_seen,
            ks=[5, 10],
            device=device,
            has_profile=val_has_prof_t,
            hist_feats=val_feats_t,
        )
        gate_str = ""
        if is_gated and epoch_gate_n > 0:
            mean_gate = epoch_gate_sum / epoch_gate_n
            gate_str = f" mean_gate={mean_gate:.4f}"
        unw_str = ""
        if is_gated and args.sample_reweight != "none":
            unw_str = f" train_loss_unw={avg_loss_unweighted:.4f}"
        aux_str = ""
        if aux_enabled and epoch_aux_n > 0:
            aux_avg = epoch_aux_loss_sum / epoch_aux_n
            g1 = (epoch_gate_y1_sum / epoch_gate_y1_n) if epoch_gate_y1_n > 0 else float("nan")
            g0 = (epoch_gate_y0_sum / epoch_gate_y0_n) if epoch_gate_y0_n > 0 else float("nan")
            aux_str = f" aux_loss={aux_avg:.4f} gate|y=1={g1:.4f} gate|y=0={g0:.4f}"
        print(
            f"[Epoch {epoch}] train_loss={avg_loss:.4f}{unw_str}{gate_str}{aux_str} "
            f"val_recall@5={val_metrics['recall@5']:.4f} "
            f"val_ndcg@5={val_metrics['ndcg@5']:.4f} "
            f"val_recall@10={val_metrics['recall@10']:.4f} "
            f"val_ndcg@10={val_metrics['ndcg@10']:.4f}"
        )

        score = val_metrics["ndcg@10"]
        if score > best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            best_metrics["epoch"] = epoch
            head_path = os.path.join(args.fusion_output_dir, "fusion_head.pt")
            ckpt = {
                "state_dict": head.state_dict(),
                "embed_dim": embed_dim,
                "head_type": args.fusion_head_type,
                "mlp_hidden": args.fusion_mlp_hidden,
            }
            if is_gated:
                ckpt["gate_features"] = gate_feature_names
                ckpt["gate_mlp_hidden"] = args.gate_mlp_hidden
                ckpt["gate_logit_init"] = args.gate_logit_init
                ckpt["item_log_pop_path"] = os.path.join(
                    args.fusion_output_dir, "item_log_pop.json"
                )
                ckpt["sample_reweight"] = args.sample_reweight
            torch.save(ckpt, head_path)
            save_json(
                os.path.join(args.fusion_output_dir, "best_val_metrics.json"), best_metrics
            )
            print(f"  new best ndcg@10={score:.4f} -> saved {head_path}")

    cfg_dump = {
        "embed_dim": embed_dim,
        "head_type": args.fusion_head_type,
        "mlp_hidden": args.fusion_mlp_hidden,
        "fusion_temperature": args.fusion_temperature,
        "fusion_num_epochs": args.fusion_num_epochs,
        "fusion_learning_rate": args.fusion_learning_rate,
        "fusion_batch_size": args.fusion_batch_size,
        "llm_profile_max_history": args.llm_profile_max_history,
        "seqrec_checkpoint": args.seqrec_checkpoint,
        "dataset_name": args.dataset_name,
    }
    if is_gated:
        cfg_dump.update({
            "gate_features": gate_feature_names,
            "gate_mlp_hidden": args.gate_mlp_hidden,
            "gate_logit_init": args.gate_logit_init,
            "gate_anchor_lambda": args.gate_anchor_lambda,
            "gate_weight_decay": args.gate_weight_decay,
            "sample_reweight": args.sample_reweight,
            "gate_aux_lambda": args.gate_aux_lambda,
            "gate_aux_alpha": args.gate_aux_alpha,
            "gate_aux_chunk_size": args.gate_aux_chunk_size,
            "gate_aux_objective": args.gate_aux_objective,
            "gate_aux_pos_weight": args.gate_aux_pos_weight,
        })
    save_json(os.path.join(args.fusion_output_dir, "fusion_config.json"), cfg_dump)
    print(f"Done. Best val ndcg@10 = {best_score:.4f}")


if __name__ == "__main__":
    main()
