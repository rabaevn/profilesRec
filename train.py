#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

from config import parse_args
from data import (
    HistoryFormatConfig,
    build_eval_catalog,
    build_examples,
    build_user_sequences,
    load_interactions,
    load_item_texts,
    to_train_dataset,
)
from evaluation import RetrievalEvalCallback, recall_and_ndcg_at_ks
from utils import print_xy_example, save_json, set_seed


def maybe_init_wandb(args: argparse.Namespace):
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None
    try:
        import wandb

        return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    except Exception as exc:
        print(f"W&B init failed ({exc}); continuing without wandb.")
        os.environ["WANDB_DISABLED"] = "true"
        return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_run = maybe_init_wandb(args)

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
    train_examples, val_examples, test_examples = build_examples(
        user_sequences=user_sequences,
        item_to_text=item_to_text,
        min_user_seq_len=args.min_user_seq_len,
        max_history_items=args.max_history_items,
        fmt=history_fmt,
    )
    print_xy_example(train_examples)

    eval_item_to_text = build_eval_catalog(
        item_to_text=item_to_text,
        interactions=interactions,
        eval_catalog=args.eval_catalog,
    )
    print(f"Evaluation catalog size: {len(eval_item_to_text):,} items (mode={args.eval_catalog})")

    if not args.eval_only:
        train_dataset = to_train_dataset(train_examples)
        model = SentenceTransformer(args.model_name, model_kwargs={"torch_dtype": "float32"})
        loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=4)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="no",
            eval_strategy="no",
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            run_name=args.wandb_run_name,
            report_to="wandb" if not args.disable_wandb else "none",
            fp16=args.fp16,
            bf16=args.bf16,
            seed=args.seed,
            data_seed=args.seed,
        )

        eval_callback = RetrievalEvalCallback(
            val_examples=val_examples,
            eval_item_to_text=eval_item_to_text,
            eval_batch_size=args.eval_batch_size,
            eval_doc_chunk_size=args.eval_doc_chunk_size,
            filter_seen_items=args.filter_seen_items,
            output_dir=args.output_dir,
            best_metric_name=args.best_metric,
            disable_wandb=args.disable_wandb,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss_fn,
            callbacks=[eval_callback],
        )
        trainer.train()
        final_step = trainer.state.global_step
    else:
        final_step = math.ceil(len(train_examples) / args.train_batch_size) * round(args.num_train_epochs)
        best_dir_check = os.path.join(args.output_dir, "best")
        if not os.path.isdir(best_dir_check):
            raise SystemExit(
                f"[eval-only] ERROR: best checkpoint not found at {best_dir_check!r}. "
                "Run training first or check --output-dir."
            )
        print(f"[eval-only] Skipping training. Will load best checkpoint from {best_dir_check}")

    best_dir = os.path.join(args.output_dir, "best")
    if os.path.isdir(best_dir):
        best_model = SentenceTransformer(best_dir)
        print(f"Loaded best checkpoint from {best_dir}")
    else:
        best_model = model
        print("Best checkpoint directory not found; using the final in-memory model.")

    final_val_metrics = recall_and_ndcg_at_ks(
        model=best_model,
        examples=val_examples,
        eval_item_to_text=eval_item_to_text,
        ks=[5, 10],
        batch_size=args.eval_batch_size,
        doc_chunk_size=args.eval_doc_chunk_size,
        filter_seen_items=args.filter_seen_items,
    )
    final_test_metrics = recall_and_ndcg_at_ks(
        model=best_model,
        examples=test_examples,
        eval_item_to_text=eval_item_to_text,
        ks=[5, 10],
        batch_size=args.eval_batch_size,
        doc_chunk_size=args.eval_doc_chunk_size,
        filter_seen_items=args.filter_seen_items,
    )

    final_dir = os.path.join(args.output_dir, "final")
    shutil.rmtree(final_dir, ignore_errors=True)
    best_model.save_pretrained(final_dir)

    save_json(os.path.join(args.output_dir, "final_val_metrics.json"), final_val_metrics)
    save_json(os.path.join(args.output_dir, "final_test_metrics.json"), final_test_metrics)

    print("\nFinal validation metrics (best checkpoint):")
    print(f"recall@5:  {final_val_metrics['recall@5']:.6f}")
    print(f"ndcg@5:    {final_val_metrics['ndcg@5']:.6f}")
    print(f"recall@10: {final_val_metrics['recall@10']:.6f}")
    print(f"ndcg@10:   {final_val_metrics['ndcg@10']:.6f}")

    print("\nFinal test metrics (best checkpoint):")
    print(f"recall@5:  {final_test_metrics['recall@5']:.6f}")
    print(f"ndcg@5:    {final_test_metrics['ndcg@5']:.6f}")
    print(f"recall@10: {final_test_metrics['recall@10']:.6f}")
    print(f"ndcg@10:   {final_test_metrics['ndcg@10']:.6f}")

    if wandb_run is not None:
        try:
            import wandb

            wandb.log({f"val/{k}": v for k, v in final_val_metrics.items()}, step=final_step)
            wandb.log({f"test/{k}": v for k, v in final_test_metrics.items()}, step=final_step)
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
