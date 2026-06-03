#!/usr/bin/env python3
from __future__ import annotations

from config import parse_profile_args
from data import (
    build_user_sequences,
    load_interactions,
    load_item_texts,
)
from profiles import (
    collect_unique_prompts,
    generate_profiles,
    generate_profiles_vllm,
    load_cf_context_cache,
)
from utils import set_seed


def main() -> None:
    args = parse_profile_args()
    set_seed(args.seed)

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

    cf_context_cache = None
    if getattr(args, "cf_context_cache", None):
        cf_context_cache = load_cf_context_cache(args.cf_context_cache)
        print(f"Loaded CF context for {len(cf_context_cache):,} keys from {args.cf_context_cache}")

    prompts = collect_unique_prompts(
        user_sequences=user_sequences,
        item_to_text=item_to_text,
        max_history=args.llm_profile_max_history,
        min_user_seq_len=args.min_user_seq_len,
        review_words=args.llm_profile_review_words,
        title_words=args.max_title_words,
        cf_context_cache=cf_context_cache,
    )
    print(f"Unique profile keys to consider: {len(prompts):,}")

    if getattr(args, "use_vllm", False):
        cache = generate_profiles_vllm(
            prompts=prompts,
            cache_path=args.llm_profile_cache,
            model_name=args.llm_profile_model,
            max_new_tokens=args.llm_profile_max_new_tokens,
            dtype=args.llm_profile_dtype,
            max_model_len=args.vllm_max_model_len,
        )
    else:
        cache = generate_profiles(
            prompts=prompts,
            cache_path=args.llm_profile_cache,
            model_name=args.llm_profile_model,
            max_new_tokens=args.llm_profile_max_new_tokens,
            batch_size=args.llm_profile_batch_size,
            dtype=args.llm_profile_dtype,
            device=args.llm_profile_device,
        )
    print(f"Profile cache now has {len(cache):,} entries at {args.llm_profile_cache}")


if __name__ == "__main__":
    main()
