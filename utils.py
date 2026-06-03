from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    return " ".join(words[:max_words]).strip()


def parse_timestamp(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    try:
        return float(s)
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_xy_example(train_examples: Sequence) -> None:
    print("\nInput/Label format:")
    print("X (History) = 'History: Item: <title_1> Item: <title_2> ...'")
    print("Y (Label)   = 'Item: <next_item_title>'")
    if not train_examples:
        return
    sample = train_examples[0]
    print("\nSample pair:")
    print(f"X: {sample.history_text}")
    print(f"Y: {sample.target_text}")
