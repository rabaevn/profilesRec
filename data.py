from __future__ import annotations

import ast
import gzip
import json
import os
import urllib.request
import zipfile
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

# MovieLens (GroupLens). Title carries the year, genres are pipe-separated.
MOVIELENS_URLS: Dict[str, str] = {
    "MovieLens-1M": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}

# Steam (McAuley/UCSD dump, python-dict gzipped JSON lines).
STEAM_URLS: Dict[str, str] = {
    "reviews": "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
    "games": "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
}

# Yelp Open Dataset is license-gated; place files manually under <root>/yelp/.
YELP_DIR = "yelp"

# Maps every supported dataset name to its loader family.
DATASET_REGISTRY: Dict[str, str] = {
    **{name: "amazon" for name in AMAZON2018_URLS},
    **{name: "movielens" for name in MOVIELENS_URLS},
    "Steam": "steam",
    "Yelp": "yelp",
}


def _family(dataset_name: str) -> str:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. Choose one of: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]


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
    family = _family(dataset_name)
    if family == "amazon":
        return _amazon_items(dataset_name, root, download_if_missing, max_title_words)
    if family == "movielens":
        return _movielens_items(dataset_name, root, download_if_missing, max_title_words)
    if family == "steam":
        return _steam_items(root, download_if_missing, max_title_words)
    if family == "yelp":
        return _yelp_items(root, max_title_words)
    raise ValueError(f"No item loader for family={family!r}")


def load_interactions(
    dataset_name: str,
    root: str,
    download_if_missing: bool,
    rating_score: float,
    valid_item_ids: Set[str],
) -> List[Interaction]:
    family = _family(dataset_name)
    if family == "amazon":
        return _amazon_interactions(dataset_name, root, download_if_missing, rating_score, valid_item_ids)
    if family == "movielens":
        interactions = _movielens_interactions(dataset_name, root, download_if_missing, rating_score, valid_item_ids)
    elif family == "steam":
        interactions = _steam_interactions(root, download_if_missing, rating_score, valid_item_ids)
    elif family == "yelp":
        interactions = _yelp_interactions(root, rating_score, valid_item_ids)
    else:
        raise ValueError(f"No interaction loader for family={family!r}")
    # Amazon SNAP files are pre-filtered to 5-core; new datasets are raw, so match it.
    return _kcore_filter(interactions, k=5)


def _kcore_filter(interactions: List[Interaction], k: int) -> List[Interaction]:
    """Iteratively drop users and items with fewer than k interactions until stable."""
    if k <= 1:
        return interactions
    kept = interactions
    while True:
        user_counts: Dict[str, int] = defaultdict(int)
        item_counts: Dict[str, int] = defaultdict(int)
        for x in kept:
            user_counts[x.user_id] += 1
            item_counts[x.item_id] += 1
        filtered = [
            x for x in kept
            if user_counts[x.user_id] >= k and item_counts[x.item_id] >= k
        ]
        if len(filtered) == len(kept):
            break
        kept = filtered
    print(f"{k}-core filtering: {len(interactions):,} -> {len(kept):,} interactions")
    return kept


def _amazon_items(
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


def _amazon_interactions(
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


def _download_and_extract_zip(url: str, dest_dir: str) -> str:
    """Download a zip (if missing) and extract it under dest_dir. Returns extraction root."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, url.split("/")[-1])
    extract_root = os.path.join(dest_dir, os.path.splitext(os.path.basename(zip_path))[0])
    if not os.path.isdir(extract_root):
        if not os.path.exists(zip_path):
            print(f"{os.path.basename(zip_path)} not found, downloading...")
            _download_file(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
    return extract_root


def _read_dat_lines(path: str) -> Iterator[List[str]]:
    """Yield '::'-delimited fields from a MovieLens .dat file (latin-1 encoded)."""
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split("::")


def _movielens_dir(dataset_name: str, root: str, download_if_missing: bool) -> str:
    ml_dir = os.path.join(os.path.abspath(root), "movielens")
    url = MOVIELENS_URLS[dataset_name]
    extract_root = os.path.join(ml_dir, os.path.splitext(url.split("/")[-1])[0])
    if not os.path.isdir(extract_root):
        if not download_if_missing:
            raise FileNotFoundError(
                f"Missing MovieLens directory: {extract_root}. Download {url} and unzip it there, "
                "or pass --download-if-missing."
            )
        extract_root = _download_and_extract_zip(url, ml_dir)
    return extract_root


def _movielens_items(
    dataset_name: str,
    root: str,
    download_if_missing: bool,
    max_title_words: int,
) -> Dict[str, str]:
    ml_dir = _movielens_dir(dataset_name, root, download_if_missing)
    movies_path = os.path.join(ml_dir, "movies.dat")
    item_to_text: Dict[str, str] = {}
    skipped = 0
    for fields in _read_dat_lines(movies_path):
        if len(fields) < 3:
            skipped += 1
            continue
        item_id, title, genres = fields[0].strip(), fields[1].strip(), fields[2].strip()
        if not item_id or not title:
            skipped += 1
            continue
        text = truncate_words(title, max_title_words)
        genres = genres.replace("|", ", ")
        if genres and genres.lower() != "(no genres listed)":
            text = f"{text} ({genres})"
        item_to_text[item_id] = f"Item: {text}"
    print(f"Loaded {len(item_to_text):,} items from metadata source={movies_path} ({skipped:,} skipped rows)")
    if not item_to_text:
        raise ValueError(f"No valid item texts from {movies_path}.")
    return item_to_text


def _movielens_interactions(
    dataset_name: str,
    root: str,
    download_if_missing: bool,
    rating_score: float,
    valid_item_ids: Set[str],
) -> List[Interaction]:
    ml_dir = _movielens_dir(dataset_name, root, download_if_missing)
    ratings_path = os.path.join(ml_dir, "ratings.dat")
    interactions: List[Interaction] = []
    skipped_missing = skipped_bad_ts = skipped_unknown_item = 0
    for fields in _read_dat_lines(ratings_path):
        if len(fields) < 4:
            skipped_missing += 1
            continue
        user_id, item_id = fields[0].strip(), fields[1].strip()
        if not user_id or not item_id:
            skipped_missing += 1
            continue
        if item_id not in valid_item_ids:
            skipped_unknown_item += 1
            continue
        try:
            rating = float(fields[2])
        except (TypeError, ValueError):
            skipped_missing += 1
            continue
        if rating < rating_score:
            continue
        ts = parse_timestamp(fields[3])
        if ts is None:
            skipped_bad_ts += 1
            continue
        interactions.append(
            Interaction(user_id=user_id, item_id=item_id, timestamp=ts, rating=rating, review_text="")
        )
    print(
        f"Loaded {len(interactions):,} interactions from source={ratings_path} "
        f"({skipped_missing:,} missing, {skipped_bad_ts:,} bad timestamps, {skipped_unknown_item:,} unknown items skipped)"
    )
    if not interactions:
        raise ValueError(f"No valid interactions loaded from {ratings_path}.")
    return interactions


def _steam_paths(root: str, download_if_missing: bool) -> Tuple[str, str]:
    steam_dir = os.path.join(os.path.abspath(root), "steam")
    os.makedirs(steam_dir, exist_ok=True)
    reviews_path = os.path.join(steam_dir, STEAM_URLS["reviews"].split("/")[-1])
    games_path = os.path.join(steam_dir, STEAM_URLS["games"].split("/")[-1])
    if download_if_missing:
        if not os.path.exists(reviews_path):
            print(f"{os.path.basename(reviews_path)} not found, downloading...")
            _download_file(STEAM_URLS["reviews"], reviews_path)
        if not os.path.exists(games_path):
            print(f"{os.path.basename(games_path)} not found, downloading...")
            _download_file(STEAM_URLS["games"], games_path)
    if not os.path.exists(reviews_path):
        raise FileNotFoundError(f"Missing Steam reviews file: {reviews_path}")
    if not os.path.exists(games_path):
        raise FileNotFoundError(f"Missing Steam games file: {games_path}")
    return reviews_path, games_path


def _steam_items(root: str, download_if_missing: bool, max_title_words: int) -> Dict[str, str]:
    _, games_path = _steam_paths(root, download_if_missing)
    item_to_text: Dict[str, str] = {}
    skipped = 0
    for row in _parse_gz_json_lines(games_path):
        item_id = row.get("id") or row.get("appid")
        name = row.get("app_name") or row.get("title")
        if item_id is None or name is None:
            skipped += 1
            continue
        item_id = str(item_id).strip()
        text = truncate_words(str(name), max_title_words)
        if not item_id or not text:
            skipped += 1
            continue
        genres = row.get("genres")
        if isinstance(genres, list) and genres:
            text = f"{text} ({', '.join(str(g) for g in genres)})"
        item_to_text[item_id] = f"Item: {text}"
    print(f"Loaded {len(item_to_text):,} items from metadata source={games_path} ({skipped:,} skipped rows)")
    if not item_to_text:
        raise ValueError(f"No valid item texts from {games_path}. Check field names ('id', 'app_name').")
    return item_to_text


def _steam_interactions(
    root: str,
    download_if_missing: bool,
    rating_score: float,
    valid_item_ids: Set[str],
) -> List[Interaction]:
    reviews_path, _ = _steam_paths(root, download_if_missing)
    interactions: List[Interaction] = []
    skipped_missing = skipped_bad_ts = skipped_unknown_item = 0
    for row in _parse_gz_json_lines(reviews_path):
        user_id = row.get("user_id") or row.get("username")
        item_id = row.get("product_id") or row.get("id")
        if user_id is None or item_id is None:
            skipped_missing += 1
            continue
        user_id, item_id = str(user_id).strip(), str(item_id).strip()
        if not user_id or not item_id:
            skipped_missing += 1
            continue
        if item_id not in valid_item_ids:
            skipped_unknown_item += 1
            continue
        ts = parse_timestamp(row.get("date"))
        if ts is None:
            skipped_bad_ts += 1
            continue
        review_text = row.get("text") if isinstance(row.get("text"), str) else ""
        interactions.append(
            Interaction(user_id=user_id, item_id=item_id, timestamp=ts, rating=0.0, review_text=(review_text or "").strip())
        )
    print(
        f"Loaded {len(interactions):,} interactions from source={reviews_path} "
        f"({skipped_missing:,} missing, {skipped_bad_ts:,} bad timestamps, {skipped_unknown_item:,} unknown items skipped)"
    )
    if not interactions:
        raise ValueError(f"No valid interactions loaded from {reviews_path}. Check field names ('user_id', 'product_id', 'date').")
    return interactions


def _open_maybe_gzip(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def _yelp_file(root: str, basename: str) -> str:
    yelp_dir = os.path.join(os.path.abspath(root), YELP_DIR)
    for cand in (os.path.join(yelp_dir, basename), os.path.join(yelp_dir, basename + ".gz")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Missing Yelp file: {os.path.join(yelp_dir, basename)}[.gz]. "
        "Yelp Open Dataset is license-gated: download it from https://www.yelp.com/dataset and place "
        f"the JSON files under {yelp_dir}/."
    )


def _yelp_items(root: str, max_title_words: int) -> Dict[str, str]:
    path = _yelp_file(root, "yelp_academic_dataset_business.json")
    item_to_text: Dict[str, str] = {}
    skipped = 0
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = row.get("business_id")
            name = row.get("name")
            if item_id is None or name is None:
                skipped += 1
                continue
            item_id = str(item_id).strip()
            text = truncate_words(str(name), max_title_words)
            if not item_id or not text:
                skipped += 1
                continue
            categories = row.get("categories")
            if isinstance(categories, str) and categories.strip():
                text = f"{text} ({categories.strip()})"
            item_to_text[item_id] = f"Item: {text}"
    print(f"Loaded {len(item_to_text):,} items from metadata source={path} ({skipped:,} skipped rows)")
    if not item_to_text:
        raise ValueError(f"No valid item texts from {path}.")
    return item_to_text


def _yelp_interactions(root: str, rating_score: float, valid_item_ids: Set[str]) -> List[Interaction]:
    path = _yelp_file(root, "yelp_academic_dataset_review.json")
    interactions: List[Interaction] = []
    skipped_missing = skipped_bad_ts = skipped_unknown_item = 0
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = row.get("user_id")
            item_id = row.get("business_id")
            if user_id is None or item_id is None:
                skipped_missing += 1
                continue
            user_id, item_id = str(user_id).strip(), str(item_id).strip()
            if not user_id or not item_id:
                skipped_missing += 1
                continue
            if item_id not in valid_item_ids:
                skipped_unknown_item += 1
                continue
            try:
                rating = float(row.get("stars")) if row.get("stars") is not None else 0.0
            except (TypeError, ValueError):
                skipped_missing += 1
                continue
            if rating < rating_score:
                continue
            ts = parse_timestamp(row.get("date"))
            if ts is None:
                skipped_bad_ts += 1
                continue
            review_text = row.get("text") if isinstance(row.get("text"), str) else ""
            interactions.append(
                Interaction(user_id=user_id, item_id=item_id, timestamp=ts, rating=rating, review_text=(review_text or "").strip())
            )
    print(
        f"Loaded {len(interactions):,} interactions from source={path} "
        f"({skipped_missing:,} missing, {skipped_bad_ts:,} bad timestamps, {skipped_unknown_item:,} unknown items skipped)"
    )
    if not interactions:
        raise ValueError(f"No valid interactions loaded from {path}.")
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
