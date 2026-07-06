# Sequential Recommendation with Sentence Transformers on Amazon 2018 Data

A modular training pipeline that fine-tunes a Sentence Transformer to predict the next item a user will interact with, given their chronologically ordered browsing/purchase history.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dependencies](#dependencies)
4. [Data](#data)
5. [Module Reference](#module-reference)
   - [utils.py](#utilspy)
   - [data.py](#datapy)
   - [config.py](#configpy)
   - [evaluation.py](#evaluationpy)
   - [train.py](#trainpy)
6. [Training Objective](#training-objective)
7. [Evaluation Methodology](#evaluation-methodology)
8. [End-to-End Pipeline](#end-to-end-pipeline)
9. [CLI Reference](#cli-reference)
10. [Output Artifacts](#output-artifacts)

---

## Overview

### What It Does

This project trains a **sequential recommendation** model. Given a user's ordered history of product interactions (e.g., purchases or reviews), the model predicts which item the user will interact with next.

The core idea is to frame sequential recommendation as a **dense retrieval** problem:

- A user's browsing history is encoded as a **query** embedding.
- Every candidate item in the catalog is encoded as a **document** embedding.
- The next item is predicted by finding the nearest item embedding to the history embedding in the shared vector space.

### Why This Approach

Traditional sequential recommendation systems typically rely on item IDs and collaborative filtering signals (e.g., matrix factorization, or autoregressive models over ID sequences). This project takes a different approach: it uses **text-based representations** of items (their product titles) and learns embeddings through **contrastive learning**.

Benefits of this approach:

- **Cold-start friendly**: New items with titles can be immediately embedded and recommended, with no interaction history required for the item itself.
- **Transfer learning**: A pre-trained language model (e.g., `all-MiniLM-L6-v2`) provides a strong initialization, encoding semantic meaning from product titles.
- **Simplicity**: No external configuration or database dependencies.

### How It Works (High Level)

1. Product titles are formatted as `"Item: <title>"` strings.
2. A user's recent history is concatenated into a single string: `"History: Item: <title_1> Item: <title_2> ..."`.
3. A Sentence Transformer is fine-tuned so that the embedding of a user's history string is close (in cosine similarity) to the embedding of the next item they will interact with.
4. Training uses **in-batch negatives**: within each batch, the positive targets of other examples serve as negative examples, requiring no explicit negative sampling.
5. At evaluation time, every item in the catalog is scored against each query, and standard retrieval metrics (Recall@K, NDCG@K) are computed.

---

## Project Structure

```
SeqRec/
  config.py                         # CLI argument parsing
  utils.py                          # Seed, timestamp parsing, text truncation, I/O helpers
  data.py                           # Data classes, download/parse, sequence building, dataset prep
  evaluation.py                     # Encoding, metrics, RetrievalEvalCallback
  train.py                          # Entry point: main(), W&B init, training loop orchestration
  environment.yml                   # Conda environment specification
  train_seqrec_st_amazon2018.py     # Legacy single-file version (retained for reference)
  DOCUMENTATION.md                  # This file
```

### Module dependency graph

```
config.py  ──> data.py (imports AMAZON2018_URLS for CLI choices)
data.py    ──> utils.py (imports parse_timestamp, truncate_words)
evaluation.py ──> data.py (imports Example), utils.py (imports save_json)
train.py   ──> config.py, data.py, evaluation.py, utils.py
```

---

## Dependencies

### Standard Library

`argparse`, `ast`, `collections`, `dataclasses`, `datetime`, `gzip`, `json`, `math`, `os`, `random`, `shutil`, `typing`, `urllib.request`

### Third-Party (Required)

| Package | Purpose |
|---|---|
| `torch` | Deep learning backend (PyTorch) |
| `numpy` | Numerical operations, matrix math for evaluation |
| `sentence-transformers` | `SentenceTransformer` model, trainer, losses, and batch samplers |
| `datasets` | HuggingFace `Dataset` for training data |
| `transformers` | `TrainerCallback` base class for custom evaluation |

### Optional

| Package | Purpose |
|---|---|
| `wandb` | Weights & Biases experiment tracking and metric logging |

A GPU is recommended for training, especially when using `--fp16` or `--bf16` mixed-precision flags.

### Environment Setup

```bash
conda env create -f environment.yml
conda activate seqrec
```

---

## Data

### Data Source

The project uses the **Amazon 2018 SNAP** dataset, hosted by the Stanford Network Analysis Project. This is a widely used benchmark for recommendation systems research. The dataset contains "5-core" data, meaning every user and every item has at least 5 interactions, ensuring a minimum density of the interaction graph.

Five product categories are supported:

| Category | Description |
|---|---|
| `Beauty` | Beauty products (default) |
| `Toys_and_Games` | Toys and games |
| `Sports_and_Outdoors` | Sports and outdoor equipment |
| `Health_and_Personal_Care` | Health and personal care products |
| `Home_and_Kitchen` | Home and kitchen products |

Each category has two data files:

- **Reviews file** (`reviews_<Category>_5.json.gz`): User reviews with ratings and timestamps.
- **Metadata file** (`meta_<Category>.json.gz`): Product metadata with titles.

The URLs for all files are stored in the `AMAZON2018_URLS` dictionary in `data.py`, mapping each category name to a `(reviews_url, meta_url)` tuple.

### Data Format

**Reviews file** -- one record per line, gzipped:

```json
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "overall": 5.0,
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```

Key fields: `reviewerID` (user ID), `asin` (product ID), `overall` (rating 1-5), `unixReviewTime` or `reviewTime` (timestamp).

**Metadata file** -- one record per line, gzipped:

```json
{
  "asin": "0000013714",
  "title": "Some Product Title Here"
}
```

Key fields: `asin` (product ID), `title` (product name).

**Important SNAP quirk**: These files are not always strict JSON. Some lines are Python dictionary literals (e.g., using single quotes). The parser handles this by trying `json.loads` first, then falling back to `ast.literal_eval`.

### Text Representation

Items and histories are converted to plain text strings for the Sentence Transformer:

- **Item text**: `"Item: <truncated title>"` -- the product title, truncated to `max_title_words` (default 20) words.
- **History text**: `"History: Item: <title_1> Item: <title_2> ... Item: <title_N>"` -- the most recent `max_history_items` (default 20) items from the user's sequence, concatenated in chronological order.

These text strings are what the model encodes into embeddings. The `"History:"` and `"Item:"` prefixes serve as lightweight structural markers, helping the model distinguish between a query (history of multiple items) and a single candidate item.

---

## Module Reference

### `utils.py`

General-purpose utilities with no project-specific dependencies.

#### `set_seed(seed)`

Sets random seeds for full reproducibility across:

- Python's `random` module
- NumPy's random number generator
- PyTorch CPU and all CUDA devices
- cuDNN (sets `deterministic=True` and `benchmark=False`)

#### `truncate_words(text, max_words)`

Splits text on whitespace and returns the first `max_words` words joined back together. Simple and fast text truncation.

#### `parse_timestamp(value)`

Converts various timestamp representations into a `float` (unix timestamp). Tries the following in order:

1. If already `int` or `float`, returns directly.
2. Tries `float(str_value)` for numeric strings.
3. Tries multiple `datetime.strptime` format strings: `%Y-%m-%d`, `%Y-%m-%d %H:%M:%S`, ISO 8601 variants with/without timezone.
4. Tries `datetime.fromisoformat` as a final fallback.
5. Returns `None` if all attempts fail.

#### `save_json(path, payload)`

Writes a dictionary to a JSON file with 2-space indentation and UTF-8 encoding.

#### `print_xy_example(train_examples)`

Prints a human-readable sample of the training data format for sanity checking:

```
X (History) = 'History: Item: <title_1> Item: <title_2> ...'
Y (Label)   = 'Item: <next_item_title>'
```

Followed by an actual sample from the training examples.

---

### `data.py`

Data classes, Amazon 2018 download/parsing, sequence construction, and dataset preparation. Imports `parse_timestamp` and `truncate_words` from `utils.py`.

#### Data Classes

##### `Interaction`

```python
@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    timestamp: float
```

Represents a single user-item interaction event (e.g., a review). Frozen (immutable) for safety and hashability.

- `user_id`: The reviewer's ID (from `reviewerID` in the reviews file).
- `item_id`: The product's ASIN (from `asin`).
- `timestamp`: Unix timestamp as a float.

##### `Example`

```python
@dataclass(frozen=True)
class Example:
    history_item_ids: Tuple[str, ...]
    history_text: str
    target_text: str
    target_item_id: str
```

Represents a single training or evaluation instance -- an (anchor, positive) pair for the contrastive loss.

- `history_item_ids`: Tuple of item IDs in the user's history up to (but not including) the target. Used during evaluation to optionally filter seen items from the ranking.
- `history_text`: The formatted history string (e.g., `"History: Item: Shampoo Item: Conditioner"`). This is the **anchor** (query) for training.
- `target_text`: The formatted target item string (e.g., `"Item: Hair Dryer"`). This is the **positive** document for training.
- `target_item_id`: The ASIN of the target item. Used for evaluation to identify the ground-truth item in the catalog.

#### Constants

##### `AMAZON2018_URLS`

A `Dict[str, Tuple[str, str]]` mapping each category name to a `(reviews_url, meta_url)` tuple. Also imported by `config.py` to populate CLI choices.

#### Internal Functions

##### `_download_file(url, dst_path)`

Downloads a file from a URL to a local path using `urllib.request.urlretrieve`. Creates parent directories automatically.

##### `_parse_gz_json_lines(path)`

A generator that reads a `.json.gz` file line by line and yields parsed dictionaries. Handles the SNAP format quirk:

1. Tries `json.loads(line)` first (standard JSON).
2. On failure, tries `ast.literal_eval(line)` (Python dict literals with single quotes, etc.).
3. Silently skips lines that can't be parsed by either method.

Only yields results that are `dict` instances.

##### `_amazon2018_paths(dataset_name, root, download_if_missing)`

Resolves local file paths for a given dataset category:

1. Looks up the review and metadata URLs from `AMAZON2018_URLS`.
2. Derives local file paths under `<root>/amazon/`.
3. If `download_if_missing=True`, downloads any missing files.
4. Raises `FileNotFoundError` if files don't exist and downloads are disabled.
5. Returns a `(reviews_path, meta_path)` tuple.

#### Public Functions

##### `load_item_texts(dataset_name, root, download_if_missing, max_title_words)`

Loads product titles from the metadata file and builds the item text catalog.

1. Gets the metadata file path via `_amazon2018_paths`.
2. Iterates over all parsed rows, extracting `asin` and `title`.
3. Truncates titles to `max_title_words` using `truncate_words`.
4. Formats each item as `"Item: <title>"`.
5. Returns `Dict[str, str]` mapping item_id to formatted text.
6. Skips rows missing `asin` or `title`, and logs the count of skipped rows.

##### `load_interactions(dataset_name, root, download_if_missing, rating_score, valid_item_ids)`

Loads user-item interactions from the reviews file.

1. Gets the reviews file path via `_amazon2018_paths`.
2. For each review row:
   - Filters out reviews below `rating_score` (default 0.0, meaning keep all).
   - Extracts `reviewerID`, `asin`, and timestamp (tries `unixReviewTime` first, falls back to `reviewTime`).
   - Skips interactions for items not in `valid_item_ids` (i.e., items without metadata).
   - Parses the timestamp via `parse_timestamp`.
3. Returns `List[Interaction]`.
4. Logs detailed skip counts (missing fields, bad timestamps, unknown items).

##### `make_history_text(history_item_ids, item_to_text, max_history_items)`

Builds the history text string from a sequence of item IDs:

1. Takes the most recent `max_history_items` IDs (slices from the end).
2. Looks up each item's text from `item_to_text`.
3. Joins them with spaces and prepends `"History: "`.

Example output: `"History: Item: Shampoo Item: Conditioner Item: Hair Mask"`

##### `build_user_sequences(interactions)`

Groups all interactions by `user_id` and sorts each user's interactions chronologically.

- Sorting key: `(timestamp, item_id)` -- the `item_id` tiebreaker ensures deterministic ordering when timestamps are identical.
- Returns `Dict[str, List[Interaction]]`.

##### `build_examples(user_sequences, item_to_text, min_user_seq_len, max_history_items)`

The core data splitting function. Creates train, validation, and test examples from user sequences using standard sequential recommendation protocols.

**Splitting strategy:**

Given a user's item sequence `[A, B, C, D, E]`:

| Split | History | Target | Protocol |
|---|---|---|---|
| **Test** | `[A, B, C, D]` | `E` | Leave-one-out (last item) |
| **Validation** | `[A, B, C]` | `D` | Leave-last-two-out (second-to-last) |
| **Train** (example 1) | `[A]` | `B` | All intermediate positions |
| **Train** (example 2) | `[A, B]` | `C` | All intermediate positions |

Rules:

- **Test**: All users with >= 2 interactions get a test example (last item as target).
- **Validation + Train**: Only users with >= `min_user_seq_len` (default 3) interactions. Validation uses the second-to-last item; training uses all positions from index 1 to N-3.
- Each example's history text is built by `make_history_text`, capped to the most recent `max_history_items`.

Returns `(train_examples, val_examples, test_examples)`.

##### `to_train_dataset(examples)`

Converts a list of `Example` objects into a HuggingFace `Dataset` with two columns:

- `"anchor"`: The history text (query).
- `"positive"`: The target item text (positive document).

This is the format expected by `CachedMultipleNegativesRankingLoss` -- it requires columns named `anchor` and `positive`.

##### `build_eval_catalog(item_to_text, interactions, eval_catalog)`

Builds the candidate item catalog used for full-ranking evaluation. Two modes:

- `"interacted"` (default): Only items that appear in at least one review. This is a smaller, faster catalog for evaluation.
- `"metadata"`: All items from the metadata file, including those with no reviews. This is a harder, more realistic evaluation scenario.

Returns `Dict[str, str]` mapping item_id to item text.

---

### `config.py`

CLI argument parsing. Imports `AMAZON2018_URLS` from `data.py` to populate dataset name choices.

#### `parse_args()`

Defines and parses all command-line arguments. Returns an `argparse.Namespace`. Arguments are grouped into dataset options, model options, sequence processing options, training hyperparameters, evaluation options, and Weights & Biases options. See the [CLI Reference](#cli-reference) section for the full list.

---

### `evaluation.py`

Encoding functions, retrieval metrics, and the training callback. Imports `Example` from `data.py` and `save_json` from `utils.py`.

#### `encode_queries(model, texts, batch_size)`

Encodes history text strings (queries) into normalized embedding vectors.

- Checks if the model has an `encode_query` method (used by asymmetric models like some bi-encoders). If so, uses it.
- Otherwise, falls back to the standard `model.encode`.
- Embeddings are L2-normalized (`normalize_embeddings=True`), so dot product equals cosine similarity.

Returns a NumPy array of shape `(num_queries, embedding_dim)`.

#### `encode_documents(model, texts, batch_size)`

Same pattern as `encode_queries`, but for item texts (documents). Checks for `encode_document` method first, then falls back to `model.encode`. Returns normalized embeddings.

The query/document distinction matters for asymmetric models where queries and documents are encoded differently (e.g., different prompt prefixes or projections). For symmetric models like `all-MiniLM-L6-v2`, both methods call the same `encode` function.

#### `recall_and_ndcg_at_ks(model, examples, eval_item_to_text, ks, batch_size, doc_chunk_size, filter_seen_items)`

The main evaluation function. Performs **full-ranking** evaluation: every candidate item in the catalog is scored against every query, and standard retrieval metrics are computed.

**Step-by-step:**

1. **Setup**: Sorts all catalog items, builds an `id_to_index` mapping, extracts query texts and ground-truth indices.

2. **Encode queries**: All history texts are encoded into a single embedding matrix.

3. **Chunked corpus encoding**: The item catalog is encoded in chunks of `doc_chunk_size` (default 4096) to avoid out-of-memory errors on large catalogs.

4. **Scoring and top-K maintenance**: For each chunk:
   - Computes cosine similarity via matrix multiplication: `query_emb @ doc_emb.T`.
   - If `filter_seen_items=True`, sets similarity scores of already-seen items to `-inf`.
   - Merges chunk results with running top-K candidates using `np.argpartition` (efficient partial sort).

5. **Metric computation**: For each query:
   - Sorts the top-K candidates by score.
   - Checks if the ground-truth item appears in the top-K.
   - **Recall@K**: 1 if ground-truth is in top-K, 0 otherwise. Averaged over all queries.
   - **NDCG@K**: `1 / log2(rank + 1)` if ground-truth is in top-K, 0 otherwise. Since there is exactly one relevant item per query, this simplifies from the general NDCG formula.

6. **Returns** a dictionary:
   ```json
   {
     "recall@5": 0.123,
     "recall@10": 0.234,
     "ndcg@5": 0.112,
     "ndcg@10": 0.198,
     "num_queries": 1000,
     "num_corpus_items": 5000
   }
   ```

#### `RetrievalEvalCallback`

```python
class RetrievalEvalCallback(TrainerCallback):
```

A custom HuggingFace `TrainerCallback` that runs full-ranking retrieval evaluation at the end of each training epoch.

##### Why a Custom Callback

The standard SentenceTransformerTrainer does not natively support full-ranking evaluation with Recall/NDCG metrics against an item catalog. This callback bridges that gap by hooking into the training loop's `on_epoch_end` event.

##### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `val_examples` | `Sequence[Example]` | Validation set examples |
| `eval_item_to_text` | `Dict[str, str]` | Candidate item catalog for ranking |
| `eval_batch_size` | `int` | Batch size for encoding |
| `eval_doc_chunk_size` | `int` | Chunk size for corpus encoding |
| `filter_seen_items` | `bool` | Whether to mask seen items |
| `output_dir` | `str` | Where to save best checkpoint |
| `best_metric_name` | `str` | Which metric to use for checkpoint selection (e.g., `ndcg@10`) |
| `disable_wandb` | `bool` | Whether W&B logging is disabled |

Internal state: `best_score` (starts at `-inf`) and `best_metrics` (the metrics dict of the best epoch).

##### `on_epoch_end` Method

Called automatically by the trainer at the end of each epoch:

1. Runs `recall_and_ndcg_at_ks` on the validation examples with K=5 and K=10.
2. Prefixes all metric keys with `val/` (e.g., `val/recall@10`).
3. Prints the metrics to stdout.
4. If the tracked metric (`best_metric_name`) exceeds the previous best:
   - Updates `best_score` and `best_metrics`.
   - Saves the model to `<output_dir>/best/` (deletes old best first).
   - Saves `best_val_metrics.json` alongside the model.
5. Logs metrics to W&B if enabled.

The `state.is_world_process_zero` guard ensures that in distributed training, only the main process saves checkpoints.

---

### `train.py`

The entry point. Imports from all other modules and orchestrates the full pipeline.

#### `maybe_init_wandb(args)`

Conditionally initializes a Weights & Biases run. If `--disable-wandb` is set, it sets the `WANDB_DISABLED` environment variable. Otherwise, it calls `wandb.init()` with the configured project name, run name, and full argument namespace as config. Gracefully falls back if W&B is not installed or initialization fails.

#### `main()`

Orchestrates the end-to-end training pipeline. See [End-to-End Pipeline](#end-to-end-pipeline) for the complete execution flow.

---

## Training Objective

### CachedMultipleNegativesRankingLoss

The pipeline uses `losses.CachedMultipleNegativesRankingLoss` from the `sentence-transformers` library (in `train.py`):

```python
loss_fn = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
```

**How it works:**

This is a **contrastive loss** that uses in-batch negatives:

- Each training batch contains N (anchor, positive) pairs.
- For each anchor, its paired positive is the correct match, and **all other positives in the batch** serve as negatives.
- The loss encourages the model to assign higher cosine similarity to the correct (anchor, positive) pair than to any (anchor, negative) pair.
- Mathematically, this is a cross-entropy loss over the similarity matrix, where the diagonal entries (correct pairs) should dominate.

**"Cached" variant:**

The "cached" version uses **gradient caching** to decouple the memory cost of the contrastive loss from the batch size. Embeddings are computed in mini-batches of `mini_batch_size=16`, cached without gradients, then the full loss is computed over the large batch. Gradients are back-propagated through the cached embeddings. This allows using large effective batch sizes (important for contrastive learning) without running out of GPU memory.

### BatchSamplers.NO_DUPLICATES

The trainer is configured with `batch_sampler=BatchSamplers.NO_DUPLICATES` (in `train.py`). This ensures that no two training examples in the same batch share the same positive text. This is critical because:

- If two examples had the same positive, using one as a negative for the other would be a **false negative**, harming training.
- It also maximizes the diversity of negatives within each batch, improving the quality of the contrastive signal.

---

## Evaluation Methodology

### Full-Ranking Protocol

Unlike some recommendation benchmarks that sample a small set of negative items (e.g., 100 random negatives), this project performs **full-ranking evaluation**: every item in the evaluation catalog is scored and ranked for each query. This is a more rigorous evaluation that better reflects real-world retrieval performance.

### Metrics

**Recall@K**: The fraction of queries where the ground-truth next item appears in the top-K ranked results.

```
Recall@K = (number of queries with hit in top-K) / (total queries)
```

**NDCG@K** (Normalized Discounted Cumulative Gain): A rank-aware metric that gives higher scores when the ground-truth item is ranked higher within the top-K.

```
NDCG@K = (1 / log2(rank + 1)) if hit in top-K, else 0
```

Since there is exactly **one relevant item per query** (the next item in the sequence), the NDCG formula simplifies to the reciprocal log of the rank.

The pipeline evaluates at K=5 and K=10.

### Memory-Efficient Chunked Encoding

For large catalogs, encoding all items at once could exhaust GPU memory. The evaluation function processes the corpus in chunks of `eval_doc_chunk_size` (default 4096):

1. Encode a chunk of items.
2. Compute similarity scores between all queries and the chunk.
3. Merge the chunk's top scores with the running top-K candidates using `np.argpartition` (O(n) partial sort, much faster than full sort).
4. Repeat for all chunks.

This approach has the same result as encoding everything at once, but uses bounded memory proportional to `chunk_size` rather than `catalog_size`.

### Seen-Item Filtering

When `--filter-seen-items` is enabled, items that appear in the user's history are masked (set to `-inf` similarity) before ranking. This creates a "next new item" prediction task, ensuring the model is evaluated on its ability to recommend items the user hasn't already interacted with.

---

## End-to-End Pipeline

The `main()` function in `train.py` orchestrates the full pipeline. Here is the complete execution flow:

### Step 1: Initialization

```
config.parse_args() -> utils.set_seed(42) -> create output directory -> train.maybe_init_wandb()
```

Command-line arguments are parsed, random seeds are set for reproducibility, and W&B is optionally initialized.

### Step 2: Load Item Catalog

```
data.load_item_texts() -> Dict[item_id, "Item: <title>"]
```

The metadata file is downloaded (if needed) and parsed. Each product's title is truncated and formatted as `"Item: <title>"`. This produces the item text catalog used throughout the pipeline.

### Step 3: Load Interactions

```
data.load_interactions() -> List[Interaction]
```

The reviews file is parsed. Reviews are filtered by minimum rating score and restricted to items that exist in the item catalog. Each valid review becomes an `Interaction` object.

### Step 4: Build User Sequences

```
data.build_user_sequences() -> Dict[user_id, List[Interaction]]
```

Interactions are grouped by user and sorted chronologically within each user. This produces ordered item sequences per user.

### Step 5: Create Train/Val/Test Splits

```
data.build_examples() -> (train_examples, val_examples, test_examples)
```

User sequences are split using leave-last-two-out (train/val) and leave-one-out (test) protocols. Training examples are created from all intermediate positions in each user's sequence. A sample (X, Y) pair is printed for verification via `utils.print_xy_example`.

### Step 6: Build Evaluation Catalog

```
data.build_eval_catalog() -> Dict[item_id, "Item: <title>"]
```

Either all metadata items or only interacted items are selected as the candidate pool for evaluation.

### Step 7: Prepare Training Dataset

```
data.to_train_dataset() -> HuggingFace Dataset {"anchor": [...], "positive": [...]}
```

Training examples are converted to a HuggingFace `Dataset` with `anchor` (history text) and `positive` (target item text) columns.

### Step 8: Load Model and Loss

```
SentenceTransformer("all-MiniLM-L6-v2")
CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
```

The pre-trained Sentence Transformer is loaded, and the contrastive loss function is initialized.

### Step 9: Configure Training

```
SentenceTransformerTrainingArguments(...)
```

Training hyperparameters are set: learning rate, batch size, warmup ratio, weight decay, gradient accumulation, precision (fp16/bf16), logging frequency. The built-in evaluation and checkpointing are disabled (`save_strategy="no"`, `eval_strategy="no"`) because the custom callback handles this.

### Step 10: Train

```
SentenceTransformerTrainer(model, args, train_dataset, loss, callbacks=[eval_callback])
trainer.train()
```

The trainer runs for `num_train_epochs` epochs. At the end of each epoch, the `RetrievalEvalCallback` runs full-ranking evaluation on the validation set and saves the best checkpoint.

### Step 11: Load Best Checkpoint

```
SentenceTransformer("<output_dir>/best/")
```

After training completes, the best checkpoint (selected by the callback based on `best_metric`) is loaded. If no checkpoint was saved (e.g., single epoch with no improvement), the final in-memory model is used.

### Step 12: Final Evaluation

```
evaluation.recall_and_ndcg_at_ks(best_model, val_examples, ...)
evaluation.recall_and_ndcg_at_ks(best_model, test_examples, ...)
```

The best model is evaluated on both the validation and test sets with full-ranking Recall@5/10 and NDCG@5/10.

### Step 13: Save Outputs

```
best_model.save_pretrained("<output_dir>/final/")
utils.save_json("final_val_metrics.json", ...)
utils.save_json("final_test_metrics.json", ...)
```

The final model is saved, metrics are written to JSON files, and results are printed to stdout. If W&B is enabled, final metrics are logged and the run is closed.

### Pipeline Diagram

```
 Download/Parse Data
        |
        v
 +------+-------+
 |               |
 v               v
Item Texts    Interactions          [data.py]
 |               |
 |               v
 |        User Sequences
 |               |
 |               v
 +----> Train/Val/Test Examples
               |
               v
     HuggingFace Dataset
     {"anchor", "positive"}
               |
               v
     SentenceTransformer            [train.py]
     + Contrastive Loss
               |
               v
         Training Loop
         (with per-epoch eval)      [evaluation.py]
               |
               v
       Load Best Checkpoint
               |
               v
   Final Eval on Val + Test         [evaluation.py]
               |
               v
   Save Model + Metrics JSON        [utils.py]
```

---

## CLI Reference

The entry point is `train.py`:

```bash
python train.py [OPTIONS]
```

### Dataset Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset-name` | str | `Beauty` | Amazon 2018 category. Choices: `Beauty`, `Health_and_Personal_Care`, `Home_and_Kitchen`, `Sports_and_Outdoors`, `Toys_and_Games` |
| `--root` | str | `./raw_data` | Root directory for downloaded SNAP files |
| `--rating-score` | float | `0.0` | Minimum rating threshold. Set to `0.0` to keep all reviews |
| `--download-if-missing` | flag | `True` | Automatically download missing data files |
| `--no-download-if-missing` | flag | -- | Disable automatic download; require files to exist locally |

### Model Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model-name` | str | `sentence-transformers/all-MiniLM-L6-v2` | Pre-trained Sentence Transformer to fine-tune |
| `--output-dir` | str | `outputs/next_item_st` | Directory for checkpoints, final model, and metrics |
| `--seed` | int | `42` | Random seed for reproducibility |

### Sequence Processing

| Argument | Type | Default | Description |
|---|---|---|---|
| `--max-title-words` | int | `20` | Max words to keep from product titles |
| `--max-history-items` | int | `20` | Max items in the history context string |
| `--min-user-seq-len` | int | `3` | Minimum interactions for a user to be included in train/val |

### Training Hyperparameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num-train-epochs` | float | `1` | Number of training epochs |
| `--learning-rate` | float | `2e-5` | Learning rate |
| `--train-batch-size` | int | `4` | Per-device training batch size |
| `--eval-batch-size` | int | `8` | Per-device evaluation batch size |
| `--warmup-ratio` | float | `0.1` | Fraction of training steps for linear warmup |
| `--weight-decay` | float | `0.01` | L2 weight decay |
| `--gradient-accumulation-steps` | int | `1` | Gradient accumulation steps |
| `--logging-steps` | int | `50` | Log training loss every N steps |
| `--fp16` | flag | `False` | Enable FP16 mixed precision |
| `--bf16` | flag | `False` | Enable BF16 mixed precision |

### Evaluation Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--eval-catalog` | str | `interacted` | Candidate catalog for ranking. `interacted`: only reviewed items. `metadata`: all metadata items |
| `--eval-doc-chunk-size` | int | `4096` | Number of documents to encode per chunk during evaluation |
| `--filter-seen-items` | flag | `False` | Mask previously seen items during ranking |
| `--best-metric` | str | `ndcg@10` | Metric for selecting the best checkpoint. Choices: `ndcg@10`, `recall@10`, `ndcg@5`, `recall@5` |

### Weights & Biases Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--wandb-project` | str | `amazon-next-item-st` | W&B project name |
| `--wandb-run-name` | str | `next-item-embedding` | W&B run name |
| `--disable-wandb` | flag | `False` | Disable W&B logging entirely |

### Example Usage

```bash
# Quick test run on Beauty dataset
python train.py --dataset-name Beauty --num-train-epochs 1

# Full training with W&B on Toys_and_Games
python train.py \
    --dataset-name Toys_and_Games \
    --num-train-epochs 3 \
    --train-batch-size 32 \
    --learning-rate 2e-5 \
    --fp16 \
    --wandb-project my-project

# Evaluate against full metadata catalog with seen-item filtering
python train.py \
    --dataset-name Beauty \
    --eval-catalog metadata \
    --filter-seen-items \
    --best-metric recall@10
```

---

## Output Artifacts

After a training run, the output directory (default `outputs/next_item_st/`) contains:

```
outputs/next_item_st/
  best/                      # Best checkpoint (saved during training)
    config.json
    model.safetensors
    ...
    best_val_metrics.json    # Validation metrics at the time of saving
  final/                     # Final model (best checkpoint re-saved after training)
    config.json
    model.safetensors
    ...
  final_val_metrics.json     # Validation metrics from the best model
  final_test_metrics.json    # Test metrics from the best model
```

Metric JSON files follow this format:

```json
{
  "recall@5": 0.0456,
  "recall@10": 0.0734,
  "ndcg@5": 0.0312,
  "ndcg@10": 0.0415,
  "num_queries": 22363,
  "num_corpus_items": 12101
}
```

The saved model in `final/` can be loaded directly with:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("outputs/next_item_st/final")
```
