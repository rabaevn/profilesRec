from __future__ import annotations

import ast
import gzip
import json
import os
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Set, Tuple

from datasets import Dataset

from utils import parse_timestamp, truncate_words


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: float
    rating: float = 0.0
    review_text: str = ""


@dataclass(frozen=True)
class Example:
    history_item_ids: Tuple[str, ...]
    history_text: str
    target_text: str
    target_item_id: str
    user_id: str


@dataclass(frozen=True)
class HistoryFormatConfig:
    time_text: bool = False
    rating_text: bool = False
    sep: bool = False
    pos_marker: bool = False


AMAZON2018_URLS: Dict[str, Tuple[str, str]] = {
    "Beauty": (
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
    ),
    "Toys_and_Games": (
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
    ),
    "Sports_and_Outdoors": (
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
    ),
    "Health_and_Personal_Care": (
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Health_and_Personal_Care.json.gz",
    ),
    "Home_and_Kitchen": (
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Home_and_Kitchen.json.gz",
    ),
}


def _download_file(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    urllib.request.urlretrieve(url, dst_path)


def _parse_gz_json_lines(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # SNAP Amazon 2018 files are often python-dict lines, not strict JSON.
                try:
                    row = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(row, dict):
                yield row


def _amazon2018_paths(dataset_name: str, root: str, download_if_missing: bool) -> Tuple[str, str]:
    if dataset_name not in AMAZON2018_URLS:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Choose one of: {sorted(AMAZON2018_URLS.keys())}")

    amazon_dir = os.path.join(os.path.abspath(root), "amazon")
    os.makedirs(amazon_dir, exist_ok=True)

    reviews_url, meta_url = AMAZON2018_URLS[dataset_name]
    reviews_path = os.path.join(amazon_dir, reviews_url.split("/")[-1])
    meta_path = os.path.join(amazon_dir, meta_url.split("/")[-1])

    if download_if_missing:
        if not os.path.exists(reviews_path):
            print(f"{os.path.basename(reviews_path)} not found, downloading...")
            _download_file(reviews_url, reviews_path)
        if not os.path.exists(meta_path):
            print(f"{os.path.basename(meta_path)} not found, downloading...")
            _download_file(meta_url, meta_path)

    if not os.path.exists(reviews_path):
        raise FileNotFoundError(f"Missing reviews file: {reviews_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    return reviews_path, meta_path


def load_item_texts(
    dataset_name: str,
    root: str,
    download_if_missing: bool,
    max_title_words: int,
) -> Dict[str, str]:
    item_to_text: Dict[str, str] = {}
    skipped = 0

    _, meta_path = _amazon2018_paths(
        dataset_name=dataset_name,
        root=root,
        download_if_missing=download_if_missing,
    )

    for row in _parse_gz_json_lines(meta_path):
        item_id = row.get("asin")
        title = row.get("title")
        if item_id is None or title is None:
            skipped += 1
            continue

        item_id = str(item_id).strip()
        title = truncate_words(str(title), max_title_words)
        if not item_id or not title:
            skipped += 1
            continue

        item_to_text[item_id] = f"Item: {title}"

    print(f"Loaded {len(item_to_text):,} items from metadata source={meta_path} ({skipped:,} skipped rows)")
    if not item_to_text:
        raise ValueError(
            "No valid item texts were loaded from Amazon2018 metadata. "
            "Check that the meta_*.json.gz file exists and contains 'asin' and 'title'."
        )
    return item_to_text


def load_interactions(
    dataset_name: str,
    root: str,
    download_if_missing: bool,
    rating_score: float,
    valid_item_ids: Set[str],
) -> List[Interaction]:
    interactions: List[Interaction] = []
    skipped_missing = 0
    skipped_bad_ts = 0
    skipped_unknown_item = 0

    reviews_path, _ = _amazon2018_paths(
        dataset_name=dataset_name,
        root=root,
        download_if_missing=download_if_missing,
    )
    for row in _parse_gz_json_lines(reviews_path):
        score_raw = row.get("overall")
        rating: float = 0.0
        if score_raw is not None:
            try:
                rating = float(score_raw)
            except (TypeError, ValueError):
                continue
            if rating < rating_score:
                continue

        user_id = row.get("reviewerID")
        item_id = row.get("asin")
        ts_value = row.get("unixReviewTime")
        if ts_value is None:
            ts_value = row.get("reviewTime")

        if user_id is None or item_id is None:
            skipped_missing += 1
            continue

        user_id = str(user_id).strip()
        item_id = str(item_id).strip()
        if not user_id or not item_id:
            skipped_missing += 1
            continue
        if item_id not in valid_item_ids:
            skipped_unknown_item += 1
            continue

        ts = parse_timestamp(ts_value)
        if ts is None:
            skipped_bad_ts += 1
            continue

        review_text = row.get("reviewText")
        if not isinstance(review_text, str) or not review_text.strip():
            review_text = row.get("summary") if isinstance(row.get("summary"), str) else ""
        review_text = (review_text or "").strip()

        interactions.append(
            Interaction(
                user_id=user_id,
                item_id=item_id,
                timestamp=ts,
                rating=rating,
                review_text=review_text,
            )
        )

    print(
        f"Loaded {len(interactions):,} interactions from source={reviews_path} "
        f"({skipped_missing:,} missing, {skipped_bad_ts:,} bad timestamps, {skipped_unknown_item:,} unknown items skipped)"
    )
    if not interactions:
        raise ValueError("No valid interactions loaded. Check your inputs and key names.")
    return interactions


def make_history_text(
    history: Sequence[Interaction],
    item_to_text: Dict[str, str],
    max_history_items: int,
    fmt: HistoryFormatConfig = HistoryFormatConfig(),
) -> str:
    recent = [x for x in history[-max_history_items:] if x.item_id in item_to_text]
    if not recent:
        return "History:"

    last_ts = recent[-1].timestamp
    pieces: List[str] = []
    for idx, inter in enumerate(recent):
        parts: List[str] = []
        if fmt.pos_marker:
            parts.append(f"[{idx + 1}]")
        if fmt.rating_text and inter.rating > 0:
            parts.append(f"★{int(inter.rating)}")
        parts.append(item_to_text[inter.item_id])
        if fmt.time_text:
            delta_days = max(0, int(round((last_ts - inter.timestamp) / 86400.0)))
            parts.append(f"({delta_days} days ago)")
        pieces.append(" ".join(parts))

    joiner = " [SEP] " if fmt.sep else " "
    return "History: " + joiner.join(pieces)


def build_user_sequences(interactions: Sequence[Interaction]) -> Dict[str, List[Interaction]]:
    by_user: Dict[str, List[Interaction]] = defaultdict(list)
    for x in interactions:
        by_user[x.user_id].append(x)
    for user_id in by_user:
        by_user[user_id].sort(key=lambda x: (x.timestamp, x.item_id))
    return by_user


def build_examples(
    user_sequences: Dict[str, List[Interaction]],
    item_to_text: Dict[str, str],
    min_user_seq_len: int,
    max_history_items: int,
    fmt: HistoryFormatConfig = HistoryFormatConfig(),
) -> Tuple[List[Example], List[Example], List[Example]]:
    train_examples: List[Example] = []
    val_examples: List[Example] = []
    test_examples: List[Example] = []

    for user_id, seq in user_sequences.items():
        seq_filtered = [x for x in seq if x.item_id in item_to_text]
        if len(seq_filtered) < 2:
            continue

        # Test is leave-one-out over all users with at least 2 interactions.
        test_pos = len(seq_filtered) - 1
        test_history = seq_filtered[:test_pos]
        test_target = seq_filtered[test_pos]
        test_examples.append(
            Example(
                history_item_ids=tuple(x.item_id for x in test_history),
                history_text=make_history_text(test_history, item_to_text, max_history_items, fmt),
                target_text=item_to_text[test_target.item_id],
                target_item_id=test_target.item_id,
                user_id=user_id,
            )
        )

        # Train/validation are leave-last-two-out over users with at least min_user_seq_len
        # (default 3).
        if len(seq_filtered) < min_user_seq_len:
            continue

        for pos in range(1, len(seq_filtered) - 2):
            history = seq_filtered[:pos]
            target = seq_filtered[pos]
            train_examples.append(
                Example(
                    history_item_ids=tuple(x.item_id for x in history),
                    history_text=make_history_text(history, item_to_text, max_history_items, fmt),
                    target_text=item_to_text[target.item_id],
                    target_item_id=target.item_id,
                    user_id=user_id,
                )
            )

        val_pos = len(seq_filtered) - 2
        val_history = seq_filtered[:val_pos]
        val_target = seq_filtered[val_pos]
        val_examples.append(
            Example(
                history_item_ids=tuple(x.item_id for x in val_history),
                history_text=make_history_text(val_history, item_to_text, max_history_items, fmt),
                target_text=item_to_text[val_target.item_id],
                target_item_id=val_target.item_id,
                user_id=user_id,
            )
        )

    print(
        "Examples (train/val: leave-last-two-out, test: leave-one-out) -> "
        f"train={len(train_examples):,} val={len(val_examples):,} test={len(test_examples):,}"
    )
    if not train_examples:
        raise ValueError("No training examples produced. Lower --min-user-seq-len or check your data.")
    if not val_examples:
        raise ValueError("No validation examples produced. Need users with at least --min-user-seq-len interactions.")
    if not test_examples:
        raise ValueError("No test examples produced. Need users with at least 2 interactions.")

    return train_examples, val_examples, test_examples


def to_train_dataset(examples: Sequence[Example]) -> Dataset:
    return Dataset.from_dict(
        {
            "anchor": [x.history_text for x in examples],
            "positive": [x.target_text for x in examples],
        }
    )


def build_eval_catalog(
    item_to_text: Dict[str, str],
    interactions: Sequence[Interaction],
    eval_catalog: str,
) -> Dict[str, str]:
    if eval_catalog == "metadata":
        return dict(item_to_text)
    interacted_item_ids = {x.item_id for x in interactions if x.item_id in item_to_text}
    return {item_id: item_to_text[item_id] for item_id in sorted(interacted_item_ids)}
