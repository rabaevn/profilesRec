from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from data import Interaction
from utils import truncate_words


@dataclass(frozen=True)
class ProfilePrompt:
    user_id: str
    history_item_ids: Tuple[str, ...]
    prompt: str


def profile_cache_key(history_item_ids: Sequence[str], max_history: int) -> str:
    """SHA1 over the truncated (last max_history) item-ID window.

    Two examples whose effective windows agree share one profile, while
    differing prefixes produce distinct keys. This is what makes the cache
    safe under variable-length history without leaking future items.
    """
    window = tuple(history_item_ids[-max_history:])
    raw = ",".join(window).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


SYSTEM_MESSAGE = (
    "You build short user profiles for a product recommender. "
    "You receive the user's recent purchases (titles, star ratings, snippets of their own "
    "reviews) and a collaborative context block listing items popular among similar shoppers. "
    "Use both signals: the purchases tell you what the user actually likes; the collaborative "
    "context tells you what items co-occur in this taste cluster.\n\n"
    "Reason step by step inside <think> ... </think> tags about (1) the product themes in the "
    "history, (2) which collaborative signals reinforce or contradict those themes, and "
    "(3) the most distinctive preference that separates the right next item from look-alikes. "
    "Then on a NEW line emit only `Profile:` followed by a 2-3 sentence preference summary "
    "for retrieval. No brand names, no marketing language. Focus on product type, function, "
    "and attributes."
)

USER_INSTRUCTION_PREFIX = (
    "Recent purchases (oldest first):\n"
)


def build_profile_prompt(
    user_id: str,
    history: Sequence[Interaction],
    item_to_text: Dict[str, str],
    max_history: int,
    review_words: int,
    title_words: int,
    cf_context: Optional[Dict[str, List[str]]] = None,
) -> str:
    recent = [x for x in history[-max_history:] if x.item_id in item_to_text]
    if not recent:
        return USER_INSTRUCTION_PREFIX + "(no recent purchases)\n"

    lines: List[str] = [USER_INSTRUCTION_PREFIX]
    for idx, inter in enumerate(recent, start=1):
        title = item_to_text.get(inter.item_id, "Item")
        if title.startswith("Item: "):
            title = title[len("Item: "):]
        # Char caps bound CJK / no-space text, which truncate_words (whitespace-split)
        # cannot — without them a single multilingual review can blow past the LLM context.
        title = truncate_words(title, title_words)[: title_words * 12]
        rating_str = f"{int(round(inter.rating))}★" if inter.rating > 0 else "no rating"
        review = truncate_words(inter.review_text or "", review_words)[: review_words * 10].replace("\n", " ").strip()
        if review:
            lines.append(f"{idx}. {title} — {rating_str} — review: \"{review}\"")
        else:
            lines.append(f"{idx}. {title} — {rating_str}")

    if cf_context:
        cooc = cf_context.get("cooc_titles") or []
        neigh = cf_context.get("neighbor_titles") or []
        if cooc or neigh:
            lines.append("")
            lines.append("Collaborative context:")
            if cooc:
                lines.append("- Items frequently bought right after items in this history:")
                for t in cooc:
                    lines.append(f"  · {t}")
            if neigh:
                lines.append("- Items popular among users with overlapping purchase histories:")
                for t in neigh:
                    lines.append(f"  · {t}")

    return "\n".join(lines) + "\n"


def collect_unique_prompts(
    user_sequences: Dict[str, List[Interaction]],
    item_to_text: Dict[str, str],
    max_history: int,
    min_user_seq_len: int,
    review_words: int,
    title_words: int,
    cf_context_cache: Optional[Dict[str, Dict[str, List[str]]]] = None,
) -> Dict[str, ProfilePrompt]:
    """Walk train/val/test prefix windows for every user, return one prompt per unique cache key.

    Mirrors the prefix logic in build_examples: every train history (positions 1..N-3),
    the val history (N-2), and the test history (N-1) for items present in item_to_text.

    If cf_context_cache is provided (keyed by the same profile_cache_key SHA1), the per-key
    CF context block is appended to each prompt.
    """
    unique: Dict[str, ProfilePrompt] = {}

    for user_id, seq in user_sequences.items():
        seq_filtered = [x for x in seq if x.item_id in item_to_text]
        n = len(seq_filtered)
        if n < 2:
            continue

        prefix_positions: List[int] = [n - 1]  # test history
        if n >= min_user_seq_len:
            prefix_positions.append(n - 2)  # val history
            prefix_positions.extend(range(1, n - 2))  # train histories

        for pos in prefix_positions:
            history = seq_filtered[:pos]
            history_ids = tuple(x.item_id for x in history)
            key = profile_cache_key(history_ids, max_history)
            if key in unique:
                continue
            cf_block = cf_context_cache.get(key) if cf_context_cache else None
            prompt = build_profile_prompt(
                user_id=user_id,
                history=history,
                item_to_text=item_to_text,
                max_history=max_history,
                review_words=review_words,
                title_words=title_words,
                cf_context=cf_block,
            )
            unique[key] = ProfilePrompt(
                user_id=user_id,
                history_item_ids=history_ids[-max_history:],
                prompt=prompt,
            )

    return unique


def load_cf_context_cache(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Load the JSONL produced by build_cf_context.py.

    Returns key -> {"cooc_titles": [...], "neighbor_titles": [...]}.
    """
    if not os.path.exists(path):
        return {}
    out: Dict[str, Dict[str, List[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if not isinstance(key, str):
                continue
            out[key] = {
                "cooc_titles": list(row.get("cooc_titles") or []),
                "neighbor_titles": list(row.get("neighbor_titles") or []),
            }
    return out


def extract_final_summary(raw: str) -> str:
    """Pull the 2-3 sentence profile out of a CoT-style response.

    The prompt asks the model to think inside <think>...</think> tags and then emit
    `Profile:` followed by the summary. We strip any think-block, then grab the text
    after the last 'Profile:' marker. Falls back to the original text if no marker
    is found (e.g. when CF context is absent and the LLM emits the summary directly).
    """
    text = raw
    # Drop any <think>...</think> blocks (greedy across newlines).
    import re
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE).strip()
    # Prefer text after the LAST 'Profile:' marker.
    lower = text.lower()
    idx = lower.rfind("profile:")
    if idx >= 0:
        text = text[idx + len("profile:"):]
    return text.strip().replace("\n", " ").strip()


def load_profile_cache(
    path: str, keep_keys: Optional[Iterable[str]] = None
) -> Dict[str, str]:
    """Load the profile cache into a dict.

    When ``keep_keys`` is given, only entries whose key is in that set are
    retained. Large caches (e.g. Steam has ~3M prefixes) hold every history
    window, but a fusion run only touches one prefix per example; filtering at
    load time keeps peak memory bounded by the examples actually used.
    """
    if not os.path.exists(path):
        return {}
    keep = set(keep_keys) if keep_keys is not None else None
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key")
            profile = row.get("profile")
            if isinstance(key, str) and isinstance(profile, str):
                if keep is None or key in keep:
                    out[key] = profile
    return out


def append_profile_cache(
    path: str,
    rows: Iterable[Tuple[str, str]],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for key, profile in rows:
            f.write(json.dumps({"key": key, "profile": profile}, ensure_ascii=False) + "\n")


def generate_profiles_vllm(
    prompts: Dict[str, ProfilePrompt],
    cache_path: str,
    model_name: str,
    max_new_tokens: int,
    dtype: str,
    flush_every: int = 256,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
) -> Dict[str, str]:
    """vLLM-backed generation. ~5-10x faster than the HF path for batched decoding."""
    from vllm import LLM, SamplingParams

    existing = load_profile_cache(cache_path)
    todo_keys = [k for k in prompts.keys() if k not in existing]
    print(f"Profile cache: {len(existing):,} present, {len(todo_keys):,} to generate (vLLM).")
    if not todo_keys:
        return existing

    llm = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    pending: List[Tuple[str, str]] = []
    # vLLM batches internally; we still chunk to checkpoint progress.
    chunk = 2048
    n_done = 0
    for start in range(0, len(todo_keys), chunk):
        batch_keys = todo_keys[start : start + chunk]
        conversations = [
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompts[k].prompt},
            ]
            for k in batch_keys
        ]
        outs = llm.chat(conversations, sampling_params=sampling, use_tqdm=False)
        for k, out in zip(batch_keys, outs):
            text = out.outputs[0].text if out.outputs else ""
            profile = extract_final_summary(text)
            pending.append((k, profile))
            existing[k] = profile
        n_done += len(batch_keys)
        if pending and len(pending) >= flush_every:
            append_profile_cache(cache_path, pending)
            pending = []
        print(f"  [{n_done:,}/{len(todo_keys):,}] generated", flush=True)

    if pending:
        append_profile_cache(cache_path, pending)
    return existing


def generate_profiles(
    prompts: Dict[str, ProfilePrompt],
    cache_path: str,
    model_name: str,
    max_new_tokens: int,
    batch_size: int,
    dtype: str,
    device: str,
    flush_every: int = 64,
) -> Dict[str, str]:
    """Run the LLM over every prompt key not already in the cache. Returns the merged cache."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    existing = load_profile_cache(cache_path)
    todo_keys = [k for k in prompts.keys() if k not in existing]
    print(f"Profile cache: {len(existing):,} present, {len(todo_keys):,} to generate.")
    if not todo_keys:
        return existing

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()

    pending: List[Tuple[str, str]] = []

    with torch.no_grad():
        for start in range(0, len(todo_keys), batch_size):
            batch_keys = todo_keys[start : start + batch_size]
            batch_texts: List[str] = []
            for k in batch_keys:
                p = prompts[k]
                messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": p.prompt},
                ]
                chat = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(chat)

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            prompt_len = enc["input_ids"].shape[1]
            gen_tokens = out[:, prompt_len:]
            decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

            for k, text in zip(batch_keys, decoded):
                profile = extract_final_summary(text)
                pending.append((k, profile))
                existing[k] = profile

            if len(pending) >= flush_every:
                append_profile_cache(cache_path, pending)
                pending = []

            done = start + len(batch_keys)
            if done % (batch_size * 10) == 0 or done == len(todo_keys):
                print(f"  [{done:,}/{len(todo_keys):,}] generated")

    if pending:
        append_profile_cache(cache_path, pending)

    return existing
