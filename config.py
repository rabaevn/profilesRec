from __future__ import annotations

import argparse

from data import AMAZON2018_URLS


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=sorted(AMAZON2018_URLS.keys()),
        default="Beauty",
        help="Amazon 2018 category name.",
    )
    parser.add_argument("--root", type=str, default="./raw_data", help="Root directory for downloaded SNAP files.")
    parser.add_argument(
        "--rating-score",
        type=float,
        default=0.0,
        help="Keep only reviews with overall >= rating-score.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        default=True,
        help="Download missing SNAP files automatically.",
    )
    parser.add_argument(
        "--no-download-if-missing",
        dest="download_if_missing",
        action="store_false",
        help="Disable automatic download and require files to exist locally.",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-title-words", type=int, default=20)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--min-user-seq-len", type=int, default=3)

    parser.add_argument(
        "--history-time-text",
        action="store_true",
        help="Append '(N days ago)' to each history item, relative to the most recent item.",
    )
    parser.add_argument(
        "--history-rating-text",
        action="store_true",
        help="Prepend '★R' (integer stars) to each history item.",
    )
    parser.add_argument(
        "--history-sep",
        action="store_true",
        help="Join history items with ' [SEP] ' instead of a single space.",
    )
    parser.add_argument(
        "--history-pos-marker",
        action="store_true",
        help="Prepend ordinal markers '[1] [2] ...' to history items (1 = oldest).",
    )
    parser.add_argument(
        "--enrich-text-input",
        action="store_true",
        help="Enable all text enrichment flags: --history-sep, --history-time-text, --history-rating-text, --history-pos-marker.",
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-doc-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--eval-catalog",
        type=str,
        choices=["interacted", "metadata"],
        default="interacted",
        help="Candidate catalog for full-ranking evaluation.",
    )
    parser.add_argument(
        "--filter-seen-items",
        action="store_true",
        help="If set, mask items already seen in the user's history during ranking.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SentenceTransformer for sequential recommendation on Amazon 2018 SNAP data with full-ranking evaluation."
    )

    _add_data_args(parser)
    _add_eval_args(parser)

    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", type=str, default="outputs/next_item_st")

    parser.add_argument("--num-train-epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument(
        "--best-metric",
        type=str,
        choices=["ndcg@10", "recall@10", "ndcg@5", "recall@5"],
        default="ndcg@10",
        help="Validation metric used to pick the best checkpoint.",
    )

    parser.add_argument("--wandb-project", type=str, default="amazon-next-item-st")
    parser.add_argument("--wandb-run-name", type=str, default="next-item-embedding")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load the best/ checkpoint and run final eval+save.",
    )

    return parser.parse_args()


def parse_cf_seqrec_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly fine-tune a SentenceTransformer encoder + CF fusion head on next-item retrieval."
    )
    _add_data_args(parser)
    _add_eval_args(parser)

    parser.add_argument(
        "--seqrec-init-checkpoint",
        type=str,
        default="outputs/seqrec_qwen3_beauty/best",
        help="Initial seqrec encoder checkpoint to continue fine-tuning from.",
    )
    parser.add_argument(
        "--user-cf-emb-path",
        type=str,
        required=True,
        help="Path to user_cf_emb_<dataset>.npz produced by build_user_cf_emb.py.",
    )
    parser.add_argument(
        "--cf-fusion-head-type",
        type=str,
        choices=["linear", "mlp", "identity_linear", "residual_gate"],
        default="residual_gate",
    )
    parser.add_argument("--cf-fusion-mlp-hidden", type=int, default=512)

    parser.add_argument("--output-dir", type=str, default="outputs/seqrec_qwen3_beauty_cf")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4,
                        help="LR for the fusion head (typically larger than the encoder LR).")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--mnr-scale", type=float, default=20.0,
                        help="MultipleNegativesRankingLoss scale (matches stock ST default).")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--best-metric",
        type=str,
        choices=["ndcg@10", "recall@10", "ndcg@5", "recall@5"],
        default="ndcg@10",
    )
    return parser.parse_args()


def parse_profile_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM user profiles from Amazon review histories. Caches one profile per (user, history-prefix) key."
    )
    _add_data_args(parser)

    parser.add_argument(
        "--llm-profile-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF causal LM used to generate user profiles (frozen, greedy decoding).",
    )
    parser.add_argument(
        "--llm-profile-cache",
        type=str,
        required=True,
        help="JSONL cache file. Resumable: keys already present are skipped.",
    )
    parser.add_argument(
        "--llm-profile-max-history",
        type=int,
        default=20,
        help="Most-recent K history items fed to the LLM. Also defines the cache key truncation.",
    )
    parser.add_argument(
        "--llm-profile-review-words",
        type=int,
        default=40,
        help="Per-item word cap on review snippets in the prompt.",
    )
    parser.add_argument(
        "--llm-profile-max-new-tokens",
        type=int,
        default=96,
        help="Generation length cap for each profile.",
    )
    parser.add_argument(
        "--llm-profile-batch-size",
        type=int,
        default=8,
        help="Batch size for LLM generation.",
    )
    parser.add_argument(
        "--llm-profile-dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="LLM dtype.",
    )
    parser.add_argument(
        "--llm-profile-device",
        type=str,
        default="cuda",
        help="Device for the LLM.",
    )
    parser.add_argument(
        "--cf-context-cache",
        type=str,
        default=None,
        help="Optional JSONL produced by build_cf_context.py. When provided, each "
             "prompt is augmented with a collaborative context block (top cooc + "
             "top neighbor items) keyed by profile_cache_key.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for generation (5-10x faster than HF transformers).",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=4096,
        help="vLLM max sequence length. Increase if CoT prompts get truncated.",
    )
    return parser.parse_args()


def parse_fusion_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fusion head on top of a frozen seqrec model + a frozen LLM profile signal."
    )
    _add_data_args(parser)
    _add_eval_args(parser)

    parser.add_argument(
        "--seqrec-checkpoint",
        type=str,
        required=True,
        help="Path to the frozen seqrec SentenceTransformer checkpoint (e.g. outputs/.../best or .../final).",
    )
    parser.add_argument(
        "--llm-profile-cache",
        type=str,
        required=True,
        help="JSONL cache file produced by generate_profiles.py.",
    )
    parser.add_argument(
        "--llm-profile-max-history",
        type=int,
        default=20,
        help="Must match what was used at profile-generation time (defines cache keys).",
    )
    parser.add_argument(
        "--fusion-output-dir",
        type=str,
        default="outputs/fusion",
        help="Where to save fusion_head.pt and metrics.",
    )
    parser.add_argument(
        "--fusion-head-type",
        type=str,
        choices=["linear", "mlp", "gated_profile"],
        default="linear",
        help="Architecture of the fusion head over concat(query_emb, profile_emb), "
             "or gated_profile (residual gate with per-example sigmoid).",
    )
    parser.add_argument(
        "--fusion-mlp-hidden",
        type=int,
        default=512,
        help="Hidden width for the MLP fusion head.",
    )
    parser.add_argument(
        "--gate-mlp-hidden",
        type=int,
        default=16,
        help="Hidden width for the gate MLP in gated_profile head.",
    )
    parser.add_argument(
        "--gate-logit-init",
        type=float,
        default=-6.0,
        help="Bias-init of the gate logit. Large negative -> gate starts near 0 (off).",
    )
    parser.add_argument(
        "--gate-features",
        type=str,
        default="history_len,has_profile,mean_history_pop,cos_q_p",
        help="Comma-separated list of gate input features. Any subset of "
             "history_len, has_profile, mean_history_pop, cos_q_p.",
    )
    parser.add_argument(
        "--gate-anchor-lambda",
        type=float,
        default=0.0,
        help="Weight on mean-gate penalty (encourages off-by-default). Try 1e-3 to 1e-2.",
    )
    parser.add_argument(
        "--gate-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay applied to gate_mlp parameters (separate group).",
    )
    parser.add_argument(
        "--gate-aux-lambda",
        type=float,
        default=0.0,
        help="Weight on the oracle-supervised BCE auxiliary loss on the gate "
             "logit. 0.0 disables. Only takes effect for gated_profile head.",
    )
    parser.add_argument(
        "--gate-aux-alpha",
        type=float,
        default=0.3,
        help="Late-fusion mixing weight used to compute the oracle label: "
             "rank gold under (1-a)*q.c + a*p.c vs under q.c. y=1 if profile-mix "
             "ranks the gold strictly better.",
    )
    parser.add_argument(
        "--gate-aux-chunk-size",
        type=int,
        default=2048,
        help="Row-chunk size for the oracle-label precompute matmul "
             "(query batch x catalog). Lower if GPU memory is tight.",
    )
    parser.add_argument(
        "--gate-aux-objective",
        type=str,
        choices=["any_uplift", "r10_recovery"],
        default="any_uplift",
        help="Oracle-label rule. any_uplift (default): y=1 if profile-mix strictly "
             "improves the gold rank vs text-only — trains soft re-ranking. "
             "r10_recovery: y=1 only if text-only misses top-10 AND profile-mix "
             "recovers it (rank<=10) — directly targets Recall@10 boundary crossings.",
    )
    parser.add_argument(
        "--gate-aux-pos-weight",
        type=str,
        choices=["none", "balanced"],
        default="none",
        help="Optional BCE pos_weight. balanced computes n_neg/n_pos from the "
             "precomputed train labels and passes it to "
             "binary_cross_entropy_with_logits. Required for sparse-positive "
             "objectives (r10_recovery) — otherwise BCE collapses to all-zero.",
    )
    parser.add_argument(
        "--sample-reweight",
        type=str,
        choices=["none", "log_hl", "sqrt_hl", "linear_hl"],
        default="none",
        help="Per-example InfoNCE reweighting by history_len. Only takes effect "
             "for --fusion-head-type gated_profile. sqrt_hl is a balanced default; "
             "linear_hl is aggressive (counteracts ~10:1 cold/heavy population skew).",
    )
    parser.add_argument("--fusion-num-epochs", type=int, default=3)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fusion-batch-size", type=int, default=512)
    parser.add_argument("--fusion-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--fusion-temperature",
        type=float,
        default=0.05,
        help="Temperature for InfoNCE over in-batch + target embeddings.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="Batch size used for the one-shot pre-encoding pass.",
    )
    parser.add_argument(
        "--fusion-device",
        type=str,
        default="cuda",
    )
    return parser.parse_args()


def parse_rerank_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot cross-encoder reranking of fusion-head top-K candidates."
    )
    _add_data_args(parser)
    _add_eval_args(parser)

    parser.add_argument("--seqrec-checkpoint", type=str, required=True)
    parser.add_argument("--llm-profile-cache", type=str, required=True)
    parser.add_argument("--llm-profile-max-history", type=int, default=20)
    parser.add_argument(
        "--fusion-head-path",
        type=str,
        required=True,
        help="Path to fusion_head.pt produced by train_fusion.py.",
    )
    parser.add_argument(
        "--fusion-head-type",
        type=str,
        choices=["linear", "mlp", "gated_profile"],
        default="linear",
    )
    parser.add_argument("--fusion-mlp-hidden", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--fusion-device", type=str, default="cuda")

    parser.add_argument(
        "--rerank-model-name",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=100,
        help="Number of bi-encoder candidates passed to the cross-encoder per query.",
    )
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--rerank-device", type=str, default="cuda")
    parser.add_argument(
        "--rerank-dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--rerank-max-length",
        type=int,
        default=4096,
        help="Tokenizer max length for the wrapped (instruction + query + document) sequence.",
    )
    parser.add_argument(
        "--rerank-instruction",
        type=str,
        default=None,
        help="Override the default cross-encoder instruction string.",
    )
    parser.add_argument(
        "--rerank-split",
        type=str,
        choices=["val", "test", "both"],
        default="val",
    )
    parser.add_argument(
        "--rerank-output-dir",
        type=str,
        required=True,
        help="Directory to write rerank_metrics.json.",
    )
    parser.add_argument(
        "--rerank-limit-queries",
        type=int,
        default=0,
        help="If >0, restrict to the first N queries (for smoke tests).",
    )
    parser.add_argument(
        "--anchor-mode",
        type=str,
        choices=["history-profile", "profile"],
        default="history-profile",
        help="Cross-encoder anchor text. 'history-profile' (default) concatenates "
             "history + profile; 'profile' uses the LLM profile only.",
    )
    parser.add_argument(
        "--rerank-lora-adapter",
        type=str,
        default=None,
        help="Path to a LoRA adapter directory (PEFT format) to load on top of the "
             "base reranker. None = zero-shot reranker.",
    )
    parser.add_argument(
        "--rerank-alpha",
        type=float,
        default=1.0,
        help="Score-fusion weight on the reranker. final = alpha * z(logit_diff) "
             "+ (1 - alpha) * z(bi_cos). 1.0 = reranker only (original behavior); "
             "0.0 = bi-encoder only.",
    )
    parser.add_argument(
        "--no-fusion-head",
        action="store_true",
        help="Bypass the fusion head and use pure text-encoder cosine (q_emb @ catalog) "
             "for top-K retrieval. The --fusion-head-path is then ignored.",
    )
    return parser.parse_args()


def parse_fusion_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained fusion head with full-ranking recall/ndcg@k."
    )
    _add_data_args(parser)
    _add_eval_args(parser)

    parser.add_argument("--seqrec-checkpoint", type=str, required=True)
    parser.add_argument("--llm-profile-cache", type=str, required=True)
    parser.add_argument("--llm-profile-max-history", type=int, default=20)
    parser.add_argument(
        "--fusion-head-path",
        type=str,
        required=True,
        help="Path to fusion_head.pt produced by train_fusion.py.",
    )
    parser.add_argument(
        "--fusion-head-type",
        type=str,
        choices=["linear", "mlp", "gated_profile"],
        default="linear",
    )
    parser.add_argument("--fusion-mlp-hidden", type=int, default=512)
    parser.add_argument(
        "--zero-profile-ablation",
        action="store_true",
        help="Replace profile embeddings with zeros. Should approximately match the seqrec baseline.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path to write metrics JSON. Defaults to <fusion-head dir>/eval_metrics.json.",
    )
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--fusion-device", type=str, default="cuda")
    return parser.parse_args()


def _add_cf_fusion_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seqrec-checkpoint",
        type=str,
        required=True,
        help="Path to the frozen seqrec SentenceTransformer checkpoint.",
    )
    parser.add_argument(
        "--user-cf-emb-path",
        type=str,
        required=True,
        help="Path to user_cf_emb_<dataset>.npz produced by build_user_cf_emb.py.",
    )
    parser.add_argument(
        "--cf-fusion-head-type",
        type=str,
        choices=["linear", "mlp", "identity_linear", "residual_gate", "learnable_residual_gate"],
        default="linear",
    )
    parser.add_argument("--cf-fusion-mlp-hidden", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--fusion-device", type=str, default="cuda")
    parser.add_argument(
        "--cf-gate-logit-init",
        type=float,
        default=-6.0,
        help="Initial value for sigmoid-gate logit in learnable_residual_gate. "
             "Large negative -> fusion starts indistinguishable from lambda0.",
    )
    parser.add_argument(
        "--cf-table-lr-scale",
        type=float,
        default=0.1,
        help="LR multiplier for the LearnableCfTable.weight parameter group. "
             "SVD init is already informative; we want fine refinement, not drift.",
    )
    parser.add_argument(
        "--cf-anchor-lambda",
        type=float,
        default=0.0,
        help="Weight on ||cf_table - cf_init||^2 anchor loss (has_cf=True users only). "
             "0 disables the anchor.",
    )


def parse_cf_fusion_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CF fusion head over concat(text_query_emb, cf_user_emb)."
    )
    _add_data_args(parser)
    _add_eval_args(parser)
    _add_cf_fusion_shared(parser)

    parser.add_argument(
        "--cf-fusion-output-dir",
        type=str,
        default="outputs/cf_fusion",
        help="Where to save cf_fusion_head.pt and metrics.",
    )
    parser.add_argument("--fusion-num-epochs", type=int, default=3)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fusion-batch-size", type=int, default=512)
    parser.add_argument("--fusion-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--fusion-temperature",
        type=float,
        default=0.05,
        help="Temperature for InfoNCE over in-batch + target embeddings.",
    )
    return parser.parse_args()


def parse_cf_fusion_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CF fusion head with full-ranking recall/ndcg@k."
    )
    _add_data_args(parser)
    _add_eval_args(parser)
    _add_cf_fusion_shared(parser)

    parser.add_argument(
        "--cf-fusion-head-path",
        type=str,
        required=True,
        help="Path to cf_fusion_head.pt produced by train_cf_fusion.py.",
    )
    parser.add_argument(
        "--zero-cf-ablation",
        action="store_true",
        help="Zero the CF embedding before it enters the head (head still applied).",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path to write metrics JSON. Defaults to <head dir>/eval_metrics.json.",
    )
    return parser.parse_args()
