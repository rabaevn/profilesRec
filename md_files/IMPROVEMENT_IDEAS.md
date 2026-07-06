# Improvement Ideas for SeqRec

Baseline reference (Beauty, 1 epoch, batch_size=4, all-MiniLM-L6-v2):
- Test Recall@10 = 5.8%, NDCG@10 = 2.6%
- Catalog: 12,094 items, 22,363 test queries

---

## 1. Training Regime (Highest Expected Impact)

### Larger Batch Sizes
The default `train_batch_size=4` is far too small for contrastive learning. In-batch negatives means each example only sees 3 negatives per step. Research consistently shows contrastive models need hundreds or thousands of negatives. Increase to 128-512 with gradient accumulation (e.g., `--train-batch-size 64 --gradient-accumulation-steps 4`). The `CachedMultipleNegativesRankingLoss` already supports this via gradient caching.

### More Epochs
Default is 1 epoch. The model is almost certainly under-trained. Run 3-10 epochs and let the best-checkpoint callback do its job. Monitor for overfitting via val metrics — the callback infrastructure is already in place.

### Temperature Tuning
The `CachedMultipleNegativesRankingLoss` has a `temperature` parameter (default 0.05 in sentence-transformers). This controls the sharpness of the softmax over similarities. Exposing this as a CLI argument and sweeping values (0.01-0.1) could meaningfully change results.

### Learning Rate & Scheduler
The default `2e-5` is reasonable but a single fixed value. Consider:
- Exposing the scheduler type (linear, cosine, cosine-with-restarts)
- Trying higher LR (5e-5, 1e-4) with cosine decay, especially with larger batch sizes

---

## 2. Hard Negative Mining

Currently, the only negatives are random in-batch items. These are mostly easy negatives (dissimilar products). Hard negatives would significantly sharpen the model's discrimination ability.

### Approaches
- **Online hard negatives**: After a warmup phase, use the current model to find near-miss items for each query and include them as explicit negatives in the training data.
- **Category-aware negatives**: Sample negatives from the same product subcategory (e.g., other shampoos when the target is a shampoo). Requires leveraging category metadata from the SNAP files.
- **Static hard negatives**: Pre-compute hard negatives with the base model before fine-tuning (a single offline pass). Sentence-transformers supports `MultipleNegativesRankingLoss` with explicit hard negatives via triplet format.

---

## 3. Richer Item Representations

### Use More Metadata Fields
Currently, only product titles are used. The Amazon metadata files also contain:
- **Category hierarchy** (e.g., `Beauty > Hair Care > Shampoo`)
- **Brand**
- **Description**
- **Price**

Enriching the item text template to something like `"Item: [title] | Category: [category] | Brand: [brand]"` gives the model more semantic signal with no architectural changes.

### Incorporate Review Text
Reviews contain rich semantic information about user preferences and item qualities. Options:
- Summarize top reviews per item and append to item representation.
- Use review text as additional positive anchors (multi-view contrastive learning).

### Rating as Signal Strength
Ratings are currently only used for filtering (below threshold). A 5-star rating indicates stronger preference than a 3-star rating. Consider:
- Weighting training examples by rating.
- Filtering training pairs to only use high-confidence (4-5 star) positive interactions.
- Using low-rated items as explicit hard negatives.

---

## 4. Sequence Modeling

### Positional / Temporal Signals
The history text is a flat concatenation with no notion of position or recency. Recent items should matter more than older ones. Ideas:
- Add position markers: `"History: [1] Item: Shampoo [2] Item: Conditioner [3] Item: Hair Mask"` where higher numbers are more recent.
- Add temporal gaps: `"Item: Shampoo (3 days ago) Item: Conditioner (1 day ago)"` to capture purchase velocity.

### Explicit Separators
Items in the history string are delimited only by spaces, which is ambiguous when titles contain spaces. Using `[SEP]` tokens or structured delimiters would make item boundaries clearer to the tokenizer.

### Data Augmentation
- **Subsequence sampling**: For a user with 10 items, randomly sample contiguous subsequences of length 3-8 as additional training examples (beyond just the fixed prefix-based examples).
- **Item dropout**: Randomly drop 10-20% of history items during training to improve robustness to incomplete histories.
- **History reversal**: Train on reversed sequences as an auxiliary task to learn bidirectional item relationships.

---

## 5. Stronger Base Models

`all-MiniLM-L6-v2` is fast but small (384 dimensions, 6 layers, 22M params). Larger models should yield better representations at the cost of training/inference speed.

### Candidates to Benchmark
| Model | Dims | Params | Notes |
|---|---|---|---|
| `all-mpnet-base-v2` | 768 | 109M | Stronger general-purpose sentence encoder |
| `BAAI/bge-base-en-v1.5` | 768 | 109M | Strong retrieval model, uses instruction prefix |
| `intfloat/e5-base-v2` | 768 | 109M | Designed for asymmetric retrieval with query/passage prompts |
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | 109M | Good general retrieval model |
| `BAAI/bge-small-en-v1.5` | 384 | 33M | Better than MiniLM at same dim, good speed/quality tradeoff |

For models that expect instruction prefixes (E5, BGE), the code's `encode_query`/`encode_document` pattern already supports this — they'd just need the right prompt template set.

---

## 6. Evaluation Improvements

### Additional Metrics
- **MRR (Mean Reciprocal Rank)**: `1/rank` of the first relevant item, averaged. More informative than binary Recall for single-relevant-item tasks.
- **Hit Rate@K**: Same as Recall@K here (single relevant item), but explicitly naming it improves clarity.
- **Coverage**: Fraction of catalog items that appear in at least one user's top-K. Measures recommendation diversity at the system level.
- **Novelty / Popularity Bias**: Measure whether the model disproportionately recommends popular items.

### Stratified Analysis
- **By user history length**: Short-history users (2-3 items) vs. long-history users (10+). Identifies where the model struggles.
- **By item popularity**: Rare items vs. popular items. Surfaces popularity bias.
- **By category**: If cross-category metadata is available, break down metrics per category.

### Statistical Rigor
- **Bootstrap confidence intervals** on all metrics (e.g., 95% CI via 1000 bootstrap samples of the test set).
- **Per-user metric distributions**: Report median and percentiles, not just mean — the mean can hide bimodal behavior (some users easy, some impossible).

---

## 7. Inference & Deployment

### Standalone Inference Script
There is no way to run recommendations without re-running the training pipeline. Add an `inference.py` script that:
1. Loads a saved model from `outputs/next_item_st/final/`.
2. Accepts a user history (list of item titles or IDs).
3. Returns top-K recommended items.

### ANN Index for Scalable Retrieval
Full brute-force scoring (query x all items) works for 12K items but won't scale to millions. Integrate an approximate nearest neighbor index:
- **FAISS** (Facebook): GPU-accelerated, supports IVF, HNSW, PQ. The most established option.
- **Hnswlib**: Pure HNSW, very fast, simpler API.
- Build the index once after training, save it alongside the model, and load it at inference time.

### Model Optimization for Production
- **ONNX export**: Convert the Sentence Transformer to ONNX for faster CPU inference.
- **Quantization**: INT8 quantization of the model for reduced memory and faster inference.
- **Dimensionality reduction**: If using 768-dim models, apply Matryoshka-style truncation or PCA to 256 dims for faster similarity search with minimal quality loss.

---

## 8. Multi-Dataset Support

### Amazon Reviews 2023
The Amazon 2018 SNAP dataset is aging (data from ~2014 and earlier). The McAuley lab released an updated [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset with newer products, richer metadata, and more categories. Adding support for this dataset would modernize the benchmark.

### Cross-Domain Datasets
Extend beyond Amazon to test generalization:
- **MovieLens**: Movie recommendation (different modality — movie titles + genres).
- **Steam**: Video game recommendation.
- **Yelp**: Local business recommendation.

Adding a dataset abstraction layer (instead of hardcoding `AMAZON2018_URLS`) would make this straightforward.

---

## 9. Advanced Modeling Techniques

### Two-Tower Architecture
Currently, queries and items share the same encoder. A two-tower setup with separate query and document encoders could let each specialize:
- The query encoder learns to compress variable-length histories.
- The document encoder learns optimal single-item representations.
- Sentence-transformers' `Asym` models support this natively.

### Multi-Task Learning
Add auxiliary objectives alongside next-item prediction:
- **Category prediction**: Predict the category of the next item (classification head).
- **Rating prediction**: Predict the rating of the next interaction (regression head).
- **Contrastive + generative**: Add a decoder head to reconstruct the target item title from the history embedding (hybrid retrieval + generation).

### Curriculum Learning
Start training with "easy" examples (users with long histories predicting popular items) and gradually introduce harder examples (short histories, rare items). This can stabilize early training and improve final performance.

### Knowledge Distillation
Train a large model (e.g., `bge-large-en-v1.5`) first, then distill it into a smaller model (`bge-small-en-v1.5`) by training the small model to match the large model's similarity scores. This gives small-model speed with large-model quality.

---

## 10. Experiment Management

### Hyperparameter Search
Integrate with **Optuna** or **Ray Tune** to automatically search over:
- Learning rate, batch size, temperature, warmup ratio
- Max history items, max title words
- Base model choice

The training loop is simple enough that wrapping it in an Optuna objective function is straightforward.

### Config File Support
Currently, all configuration is via CLI args. Add support for YAML/JSON config files (load a config file, override with CLI args). This makes experiment reproducibility easier — just save and share the config file.

### Experiment Comparison Dashboard
Metrics are saved as flat JSON files with no systematic naming. Consider:
- Naming output directories by experiment hash or timestamp (e.g., `outputs/beauty_bge-small_bs128_ep5_20260405/`).
- A lightweight comparison script that reads all `final_test_metrics.json` files from `outputs/*/` and produces a summary table.

---

## 11. Testing & CI

### Unit Tests
The project has zero tests. Key areas to cover:
- `truncate_words`, `parse_timestamp` (pure functions, easy to test)
- `build_examples` splitting logic (verify leave-one-out / leave-last-two-out correctness on small synthetic data)
- `make_history_text` truncation and formatting
- `recall_and_ndcg_at_ks` with known inputs/outputs (e.g., gold item at rank 1, rank 5, not in top-K)
- End-to-end smoke test: train 1 step on tiny synthetic data, verify output files are created

### CI Pipeline
Add GitHub Actions to run:
- Linting (`ruff` or `flake8`)
- Unit tests on every push
- A fast smoke-test training run on a tiny synthetic dataset

---

## 12. Code Quality

### Type Checking
The code has type annotations but no `mypy` or `pyright` configuration. Adding strict type checking would catch bugs early, especially around the dict-heavy data pipeline.

### Logging
All output is via `print()`. Switching to Python's `logging` module would allow:
- Log levels (DEBUG for verbose data stats, INFO for progress, WARNING for skipped items)
- Structured logging for machine-parseable output
- Silencing output in tests

### Dependency Pinning
`environment.yml` pins torch but leaves `sentence-transformers`, `transformers`, `datasets`, and `accelerate` unpinned. These libraries have breaking API changes between versions. Pin all dependencies or add a `requirements.txt` with exact versions.

---

## Priority Ranking

Ordered by expected impact-to-effort ratio:

1. **Increase batch size + epochs** (config change only, likely 2-3x metric improvement)
2. **Try stronger base models** (config change only, `--model-name bge-small-en-v1.5`)
3. **Expose temperature as a CLI arg** (tiny code change, potentially large impact)
4. **Richer item text** (small code change in `load_item_texts`, uses existing metadata)
5. **Add positional markers to history** (small code change in `make_history_text`)
6. **Hard negative mining** (moderate code, significant impact)
7. **Add MRR metric** (small code change in `evaluation.py`)
8. **Standalone inference script** (needed for any practical use)
9. **Unit tests** (safety net for all future changes)
10. **Hyperparameter search with Optuna** (wraps existing pipeline)
11. **ANN index for scalable retrieval** (needed if catalog grows)
12. **Two-tower / advanced modeling** (higher effort, uncertain incremental gain)
