#!/usr/bin/env python3
"""Two extra cohort axes derived purely from `interactions` (no GPU/encoding):

- user_activity   : total interactions for the example's user across the full
                    dataset. Distinct from history_len (the eval-prefix length).
- target_first_ts : first-interaction timestamp of the target item. Lower = the
                    item appeared earlier (established); higher = newer item.

Both are functions of (user_id) / (target_item_id) only, so they can be baked
into a per-example parquet at generation time *or* back-filled onto an existing
parquet by reloading interactions for its dataset.

Bucket boundaries are tertiles computed on the per-example distribution, matching
the data-driven quantile convention used elsewhere (evaluate_slices / slice_fusion).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ACTIVITY_LABELS = ["light", "medium", "heavy"]          # low -> high user_activity
AGE_LABELS = ["established", "mid", "new"]               # low -> high target_first_ts


def add_activity_age_columns(df: pd.DataFrame, interactions) -> pd.DataFrame:
    """Return a copy of `df` with `user_activity` and `target_first_ts` columns.

    `df` must have `user_id` and `target_item_id`. `interactions` is the list of
    Interaction(user_id, item_id, timestamp) from data.load_interactions.
    """
    user_total: Counter = Counter(x.user_id for x in interactions)
    item_first_ts: Dict[str, float] = {}
    for x in interactions:
        cur = item_first_ts.get(x.item_id)
        if cur is None or x.timestamp < cur:
            item_first_ts[x.item_id] = x.timestamp

    out = df.copy()
    out["user_activity"] = (
        out["user_id"].map(lambda u: user_total.get(u, 0)).astype("int64")
    )
    out["target_first_ts"] = (
        out["target_item_id"].map(lambda i: item_first_ts.get(i, np.nan)).astype("float64")
    )
    return out


def backfill_from_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    root: str = "./raw_data",
    rating_score: float = 0.0,
    download_if_missing: bool = True,
    max_title_words: int = 20,
) -> pd.DataFrame:
    """Add the two columns to `df` by reloading interactions for `dataset_name`.

    Used to enrich legacy parquets that predate these columns. Imports data lazily
    so the pure-pandas helpers above stay import-light.
    """
    from data import load_interactions, load_item_texts

    item_to_text = load_item_texts(
        dataset_name=dataset_name,
        root=root,
        download_if_missing=download_if_missing,
        max_title_words=max_title_words,
    )
    interactions = load_interactions(
        dataset_name=dataset_name,
        root=root,
        download_if_missing=download_if_missing,
        rating_score=rating_score,
        valid_item_ids=set(item_to_text.keys()),
    )
    return add_activity_age_columns(df, interactions)


def _tertile_edges(values: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(v, 1.0 / 3.0)), float(np.quantile(v, 2.0 / 3.0))


def activity_buckets(series: pd.Series) -> Tuple[pd.Series, Tuple[float, float]]:
    """Map user_activity -> light/medium/heavy by tertiles. Returns (labels, edges)."""
    t1, t2 = _tertile_edges(series.to_numpy())

    def b(x: float) -> str:
        if x <= t1:
            return "light"
        if x <= t2:
            return "medium"
        return "heavy"

    return series.map(b), (t1, t2)


def age_buckets(series: pd.Series) -> Tuple[pd.Series, Tuple[float, float]]:
    """Map target_first_ts -> established/mid/new by tertiles. Returns (labels, edges).

    Lower first_ts = item appeared earlier = 'established'; NaN -> 'unknown'.
    """
    t1, t2 = _tertile_edges(series.to_numpy())

    def b(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "unknown"
        if x <= t1:
            return "established"
        if x <= t2:
            return "mid"
        return "new"

    return series.map(b), (t1, t2)
