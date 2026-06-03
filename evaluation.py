from __future__ import annotations

import math
import os
import shutil
from typing import Dict, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import TrainerCallback

from data import Example
from utils import save_json


def encode_queries(model: SentenceTransformer, texts: Sequence[str], batch_size: int) -> np.ndarray:
    if hasattr(model, "encode_query"):
        return model.encode_query(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def encode_documents(model: SentenceTransformer, texts: Sequence[str], batch_size: int) -> np.ndarray:
    if hasattr(model, "encode_document"):
        return model.encode_document(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def recall_and_ndcg_at_ks(
    model: SentenceTransformer,
    examples: Sequence[Example],
    eval_item_to_text: Dict[str, str],
    ks: Sequence[int],
    batch_size: int,
    doc_chunk_size: int,
    filter_seen_items: bool,
) -> Dict[str, float]:
    if not examples:
        raise ValueError("Need at least one example for evaluation.")
    unique_ks = sorted({int(k) for k in ks if int(k) > 0})
    if not unique_ks:
        raise ValueError("ks must contain positive integers.")

    item_ids = sorted(eval_item_to_text.keys())
    corpus_texts = [eval_item_to_text[item_id] for item_id in item_ids]
    if not corpus_texts:
        raise ValueError("Evaluation catalog is empty.")
    id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    queries = [x.history_text for x in examples]
    gold_indices = [id_to_index[x.target_item_id] for x in examples]
    query_emb = encode_queries(model, queries, batch_size=batch_size).astype(np.float32, copy=False)

    num_queries = len(examples)
    max_k = min(max(unique_ks), len(item_ids))
    top_scores = np.full((num_queries, max_k), -np.inf, dtype=np.float32)
    top_indices = np.full((num_queries, max_k), -1, dtype=np.int64)

    seen_index_arrays: Optional[List[np.ndarray]] = None
    if filter_seen_items:
        seen_index_arrays = []
        for ex in examples:
            seen_idx = [id_to_index[item_id] for item_id in ex.history_item_ids if item_id in id_to_index]
            seen_index_arrays.append(np.array(sorted(set(seen_idx)), dtype=np.int64))

    for start in range(0, len(corpus_texts), doc_chunk_size):
        end = min(start + doc_chunk_size, len(corpus_texts))
        doc_emb = encode_documents(model, corpus_texts[start:end], batch_size=batch_size).astype(np.float32, copy=False)
        chunk_scores = np.matmul(query_emb, doc_emb.T)

        if seen_index_arrays is not None:
            for q_idx, seen_idx in enumerate(seen_index_arrays):
                if seen_idx.size == 0:
                    continue
                local = seen_idx[(seen_idx >= start) & (seen_idx < end)] - start
                if local.size > 0:
                    chunk_scores[q_idx, local] = -np.inf

        chunk_indices = np.tile(np.arange(start, end, dtype=np.int64), (num_queries, 1))

        merged_scores = np.concatenate([top_scores, chunk_scores], axis=1)
        merged_indices = np.concatenate([top_indices, chunk_indices], axis=1)
        keep = np.argpartition(-merged_scores, kth=max_k - 1, axis=1)[:, :max_k]
        row_ids = np.arange(num_queries)[:, None]
        top_scores = merged_scores[row_ids, keep]
        top_indices = merged_indices[row_ids, keep]

    recalls = {k: 0.0 for k in unique_ks}
    ndcgs = {k: 0.0 for k in unique_ks}

    for i, gold_idx in enumerate(gold_indices):
        order = np.argsort(-top_scores[i])
        ranked = top_indices[i][order]

        hit_rank: Optional[int] = None
        for rank, idx in enumerate(ranked, start=1):
            if idx == gold_idx:
                hit_rank = rank
                break

        for k in unique_ks:
            effective_k = min(k, len(ranked))
            if hit_rank is not None and hit_rank <= effective_k:
                recalls[k] += 1.0
                ndcgs[k] += 1.0 / math.log2(hit_rank + 1)

    denom = float(len(examples))
    metrics: Dict[str, float] = {}
    for k in unique_ks:
        metrics[f"recall@{k}"] = recalls[k] / denom
        metrics[f"ndcg@{k}"] = ndcgs[k] / denom
    metrics["num_queries"] = denom
    metrics["num_corpus_items"] = float(len(item_ids))
    return metrics


class RetrievalEvalCallback(TrainerCallback):
    def __init__(
        self,
        val_examples: Sequence[Example],
        eval_item_to_text: Dict[str, str],
        eval_batch_size: int,
        eval_doc_chunk_size: int,
        filter_seen_items: bool,
        output_dir: str,
        best_metric_name: str,
        disable_wandb: bool,
    ) -> None:
        super().__init__()
        self.val_examples = list(val_examples)
        self.eval_item_to_text = dict(eval_item_to_text)
        self.eval_batch_size = eval_batch_size
        self.eval_doc_chunk_size = eval_doc_chunk_size
        self.filter_seen_items = filter_seen_items
        self.output_dir = output_dir
        self.best_metric_name = best_metric_name
        self.disable_wandb = disable_wandb
        self.best_score = -float("inf")
        self.best_metrics: Optional[dict] = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        metrics = recall_and_ndcg_at_ks(
            model=model,
            examples=self.val_examples,
            eval_item_to_text=self.eval_item_to_text,
            ks=[5, 10],
            batch_size=self.eval_batch_size,
            doc_chunk_size=self.eval_doc_chunk_size,
            filter_seen_items=self.filter_seen_items,
        )
        metrics = {f"val/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = float(state.epoch or 0.0)
        score = float(metrics[f"val/{self.best_metric_name}"])

        print(
            f"\n[Epoch {metrics['epoch']:.2f}] "
            f"val_recall@5={metrics['val/recall@5']:.6f} "
            f"val_ndcg@5={metrics['val/ndcg@5']:.6f} "
            f"val_recall@10={metrics['val/recall@10']:.6f} "
            f"val_ndcg@10={metrics['val/ndcg@10']:.6f}"
        )

        if score > self.best_score:
            self.best_score = score
            self.best_metrics = dict(metrics)
            if getattr(state, "is_world_process_zero", True):
                best_dir = os.path.join(self.output_dir, "best")
                os.makedirs(self.output_dir, exist_ok=True)
                shutil.rmtree(best_dir, ignore_errors=True)
                model.save_pretrained(best_dir)
                save_json(os.path.join(best_dir, "best_val_metrics.json"), metrics)
                print(f"New best checkpoint saved to {best_dir} using {self.best_metric_name}={score:.6f}")

        if not self.disable_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
            except Exception:
                pass

        return control
