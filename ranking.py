"""Single shared rank-computation path for all eval scripts.

Why this module exists: evaluate_slices.py used to compute the gold item's
score via a separate numpy elementwise path while all catalog scores came from
a BLAS matmul. The two float32 reduction orders round differently (~1e-7), so
on ~41% of Beauty test rows the gold's matmul score rounded strictly above its
separately-computed score and the gold counted *against itself* — rank inflated
by exactly 1 (Beauty test R@10 0.0949 vs the true 0.0974). Gathering the gold
from the same score matrix makes self-counting impossible; sharing one code
path (and one set of cached embeddings) across evaluate_slices.py and
slice_fusion.py makes their ranks bit-identical.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

# Rank equality across scripts requires bit-identical matmuls; TF32 would let
# GPU generation / kernel selection reintroduce precision drift.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# The query-chunk size is fixed here, NOT caller-configurable: cuBLAS picks
# shape-dependent kernels/reduction orders, so ranking the same embeddings at
# chunk 4096 vs 2048 flips near-ties between duplicate item texts (sports:
# 9 test rows). One shape everywhere = one rounding everywhere.
_RANK_CHUNK = 2048

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.ascontiguousarray(x)).float()


@torch.no_grad()
def per_example_ranks(
    query_emb: ArrayLike,
    catalog_emb: ArrayLike,
    gold_indices: ArrayLike,
    seen_indices: Optional[Sequence[ArrayLike]] = None,
    device: str = "cuda",
) -> np.ndarray:
    """1-indexed rank of each query's gold item; ties broken pessimistically
    (count of strictly-greater scores, plus 1).

    The gold score is gathered from the same score matrix as every other score
    — never recomputed through a different reduction path.

    Masking convention (decided 2026-07-19): seen items are masked to -inf,
    but the gold target is ALWAYS rankable — its score is gathered from the
    score matrix before masking, so a repeat target (gold already in the
    user's history) can still be a hit. This is the standard leave-one-out
    convention and makes the full test set meaningful on domains with repeat
    interactions (Steam: 12.8% of examples). The gold's own (possibly masked)
    entry can never count against itself: -inf is never strictly greater than
    the gathered gold score.
    """
    if not torch.cuda.is_available():
        device = "cpu"
    query_t = _to_tensor(query_emb)
    catalog_dev = _to_tensor(catalog_emb).to(device)
    gold_t = torch.as_tensor(
        gold_indices.cpu().numpy() if isinstance(gold_indices, torch.Tensor)
        else np.asarray(gold_indices),
        dtype=torch.long,
    )

    n_q = query_t.shape[0]
    ranks = np.zeros(n_q, dtype=np.int64)
    for s in range(0, n_q, _RANK_CHUNK):
        e = min(s + _RANK_CHUNK, n_q)
        q = query_t[s:e].to(device)
        scores = q @ catalog_dev.T  # (b, C)
        # Gather gold BEFORE masking (gold always rankable, even if seen).
        gold = gold_t[s:e].to(device)
        gold_scores = scores.gather(1, gold.view(-1, 1))
        if seen_indices is not None:
            for j, idx in enumerate(seen_indices[s:e]):
                idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long) \
                    if not isinstance(idx, torch.Tensor) else idx
                if idx_t.numel() > 0:
                    scores[j, idx_t.to(device)] = float("-inf")
        higher = (scores > gold_scores).sum(dim=1)
        ranks[s:e] = (higher + 1).cpu().numpy()
    return ranks
