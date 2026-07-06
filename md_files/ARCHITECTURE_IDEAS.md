# Architecture Ideas: Temporal + Rating Signal Injection

## The Core Problem

Phase 2 and Phase 3 each solve half the problem:

| | Cross-item attention | Learned temporal/rating signal |
|---|---|---|
| Phase 2 (concat text → Qwen) | ✅ full token-level | ❌ only as raw text ("30d") |
| Phase 3 (per-item → Qwen → pool) | ❌ none | ✅ learned embeddings, added post-hoc |

The goal: get both at the same time.

---

## What "cross-item attention" means here

In Phase 2, Qwen's self-attention lets every token see every other token across all items:

```
"OPI"(item1) can directly attend to "OPI"(item2) inside Qwen's layers
```

In Phase 3, items are encoded independently. They never see each other inside Qwen.

---

## Version A — Cross-Attention BEFORE Qwen's Transformer

### Idea

Enrich the token embeddings with learned temporal and rating signals **before** Qwen's
transformer layers run. This way the temporal/rating signal propagates through all 28
attention layers inside Qwen, not just added on top afterward.

The concatenated text already includes rating and temporal as readable text (★5, 30 days ago)
because those helped in Phase 2. The learned embeddings add a second, more precise signal on top:

| Signal | Format | Where Qwen sees it |
|---|---|---|
| Rating (coarse) | "★5" in the text string | reads it as a word |
| Temporal (coarse) | "(30 days ago)" in the text string | reads it as words |
| Rating (precise) | `rating_emb[5]` learned vector | injected before transformer |
| Temporal (precise) | `temporal_emb[bin(30d)]` learned vector | injected before transformer |

### Pipeline

```
Full concatenated text (with rating + temporal as text, like Phase 2 best config):
"[1] ★5 OPI Nail Lacquer (30 days ago) [SEP] [2] ★4 OPI Red Shatter (10 days ago) [SEP] [3] ★3 Revlon Top Coat (0 days ago)"

Step 1 — Vocab lookup (NOT the full transformer, just the embedding table):
  token_embs = embed_table[token_ids]          shape: (seq_len, D)

Step 2 — Learned structured embeddings (one per item, not per token):
  temporal_embs = temporal_emb[delta_days_bins]   shape: (N, D)
  rating_embs   = rating_emb[star_ratings]         shape: (N, D)
  structured = cat(temporal_embs, rating_embs)     shape: (2N, D)

Step 3 — Cross-attention (token_embs attend to structured signals):
  Q = token_embs         (seq_len, D)   ← what we want to enrich
  K = structured         (2N, D)        ← source of temporal/rating info
  V = structured         (2N, D)

  attn_out = CrossAttention(Q, K, V)    (seq_len, D)

Step 4 — Residual (never lose the original token signal):
  enriched_tokens = token_embs + attn_out        (seq_len, D)

Step 5 — Feed into Qwen's transformer layers:
  → layer 1 → layer 2 → ... → layer 28
  → pool (last token)
  → query (D,)
```

### Why the cross-attention works here

Each token can learn to "ask" the temporal/rating embeddings for information.
Tokens from item 1 will naturally attend more strongly to item 1's temporal/rating embedding.
We can also enforce this with an attention mask (item 1 tokens can only attend to item 1's
structured embeddings) — gives the model an explicit item-boundary inductive bias.

### Why this is more powerful than Version B

The temporal/rating signal enters at layer 0 and propagates through all 28 Qwen layers.
By the time the model computes cross-item attention in layer 10 (for example):
```
"OPI"(item1, 30d ago, ★5) attending to "OPI"(item2, 10d ago, ★4)
```
Both tokens already carry temporal/rating context. The attention weight between them is
informed by both the word meaning AND the time/rating difference.

### Memory

One Qwen forward pass per user (not N passes like Phase 3).
- Phase 3 at bs=8: 8 users × 20 items = 160 Qwen forward passes → ~42GB
- Version A at bs=8: 8 users × 1 Qwen forward pass (longer sequence) → estimated ~15–20GB
- Enables much larger batch sizes (bs=32 or bs=64 likely feasible)

### Implementation challenge

Need to split Qwen's forward pass into:
  1. `embed_tokens(input_ids)` → token embeddings
  2. inject cross-attention output
  3. `transformer_layers(enriched_tokens)` → run the rest

SentenceTransformer wraps HuggingFace, so access is via:
`model.backbone[0].auto_model`

Need to identify item spans from [SEP] token positions.

---

## Version B — Cross-Attention AFTER Qwen's Transformer

### Idea

Run Qwen on the full concatenated text (Phase 2 style) to get one sentence embedding.
Then use cross-attention to combine that embedding with learned temporal/rating signals.
Add the sentence embedding as a residual to ensure the original signal is never lost.

Same as Version A: the concatenated text already includes rating and temporal as readable
text (★5, 30 days ago) because those helped in Phase 2. The learned embeddings are a
second, more precise signal added on top after Qwen finishes.

### Pipeline

```
Full concatenated text (with rating + temporal as text, like Phase 2 best config):
"[1] ★5 OPI Nail Lacquer (30 days ago) [SEP] [2] ★4 OPI Red Shatter (10 days ago) [SEP] [3] ★3 Revlon Top Coat (0 days ago)"

Step 1 — Full Qwen forward pass (Phase 2):
  full_emb = qwen(concat_text)                  shape: (D,)
  (this captures cross-item meaning at token level, no temporal signal)

Step 2 — Learned structured embeddings (one per item):
  temporal_embs = temporal_emb[delta_days_bins]   shape: (N, D)
  rating_embs   = rating_emb[star_ratings]         shape: (N, D)
  structured = cat(temporal_embs, rating_embs)     shape: (2N, D)

Step 3 — Cross-attention (full_emb attends to temporal/rating):
  Q = full_emb.unsqueeze(0)     (1, D)    ← the full history signal
  K = structured                (2N, D)   ← structured per-item signals
  V = structured                (2N, D)

  attn_out = CrossAttention(Q, K, V)      (1, D)

Step 4 — Residual (critical: preserve the full Phase 2 signal):
  query = full_emb + attn_out             (D,)

Step 5 — L2 normalize → query
```

### Why the residual matters

At initialization, `attn_out` is random noise. With the residual:
```
query ≈ full_emb + small_noise ≈ full_emb
```
Training starts from Phase 2 baseline performance and improves from there.
Without the residual, training starts from random and is unstable.

### What cross-attention learns here

The model learns to ask: "given the full meaning of this history, which temporal and rating
signals are most relevant to adjust my prediction?"

Example: a user who bought 5 items all 2 years ago vs. 1 item yesterday — the attention
learns to weight recent-item temporal signals more heavily when the history is sparse but recent.

### Memory

Same as Phase 2. One Qwen forward pass per user.
- At bs=8: one 600-token forward pass × 8 users → ~15–20GB (similar to Version A)
- Backbone still fine-tunes on this path

### Implementation complexity

Clean — no hooking into Qwen internals.
Add a `CrossAttentionCombiner` module that sits entirely after `_backbone_encode`.

---

---

## Sentiment / Review Embedding Enrichment

### The Problem with Ratings Alone

★1–★5 is a single integer. It loses everything in the review text:

```
★4  "Best nail polish I've tried, lasts 2 weeks, love the brush"
★4  "Decent color but chips after one day, disappointed with quality"
```

Same rating. Completely different signal about what this user cares about and what
they'll look for next.

### Two Variants

#### Variant 1 — Sentiment Score (lightweight)

Run a small sentiment classifier on the review text → one continuous score per item.

```
review: "lasted 2 weeks, love the brush"  →  score = 0.92
review: "chips after one day"             →  score = 0.31
```

Use as a more precise pooling weight (replaces ★ rating) or as a binned embedding:
`sentiment_emb[bin(0.92)]`. Small models: VADER (CPU, rule-based) or DistilBERT (67M params).

#### Variant 2 — Review Embedding (more powerful)

Encode the full review text through a model → a dense vector per item.

```
"Best nail polish I've tried, lasts 2 weeks, love the brush"
  → [model] → review_emb  (D,)
```

This captures which aspects the user paid attention to, their vocabulary, their depth of
feeling — things the item title alone ("OPI Nail Lacquer") never encodes.

The item title embedding answers: **what is this product?**
The review embedding answers: **why did this specific user like or dislike it?**

### Data Requirement

Review text is NOT currently in the pipeline. `StructuredExample` only stores title,
rating, and delta_days. The Amazon 2018 reviews file has `reviewText` and `summary` fields
that are currently loaded but discarded in `load_interactions()`.

Changes needed:
1. Add `review_text: str` field to `Interaction` and `StructuredExample`
2. Load `reviewText` (or `summary`) from the reviews JSON
3. Pre-compute review embeddings offline and cache to disk

### Pre-Computing: Zero Runtime Cost

Review text never changes. Compute embeddings once before training, save to disk,
load as a lookup table. During training it is a single tensor index — same cost as
`temporal_emb`.

```
# offline (run once):
for user, item in all_interactions:
    review_emb[user][item] = sentiment_model(review_text)
torch.save(review_emb, "review_embs_cache.pt")

# training (instant lookup):
review_emb_i = review_emb_cache[user_id][item_id]
```

### Where It Fits in the Pipeline

Review embedding is a per-item feature, same shape as temporal/rating. It plugs into
both Version A and Version B as an additional input to the cross-attention:

```
Version B — structured signals grow from (2N, D) to (3N, D):

  K = cat(temporal_embs,    (N, D)
          rating_embs,      (N, D)
          review_embs)      (N, D)   ← new
    → shape (3N, D)

  cross-attention(Q=full_emb, K, V) → combined
  + residual → query
```

Version A is identical — review_embs joins temporal/rating in the structured embeddings
that are injected into token embeddings before Qwen's layers.

### Which Sentiment Model to Use

| Model | Size | Cost | Signal |
|---|---|---|---|
| VADER | ~1MB, CPU | negligible | positive/negative/neutral score only |
| DistilBERT-SST2 | 67M params | pre-compute once | richer sentiment, embeddings extractable |
| Qwen3-Embedding-0.6B (same backbone) | 600M params | pre-compute once | richest — same embedding space as item titles |

Using the same backbone (Qwen) to encode reviews keeps everything in the same 1024-dim
space. No projection needed. Review embeddings are directly comparable to item title
embeddings and query embeddings.

---

## Comparison

```
                     token level    learned temporal     review signal   memory      implementation
                     cross-item     depth inside Qwen                    estimate    complexity
Phase 2              ✅             ❌ (text only)        ❌              ~15GB       none (baseline)
Phase 3              ❌             ❌ (post-hoc add)     ❌              ~42GB       low
Version A            ✅             ✅ all 28 layers       optional ✅     ~15–20GB    high
Version B            ✅             ❌ (post-hoc)          optional ✅     ~15–20GB    medium
```

---

## Design Principles (apply to all versions)

### 1. Always use residual connections on new modules
```
output = new_module(input) + input
```
Guarantees the model starts at baseline performance. New module learns incrementally.

### 2. Zero-initialize the output projection of any new module
```python
nn.init.zeros_(new_module.out_proj.weight)
nn.init.zeros_(new_module.out_proj.bias)
```
At step 0: `new_module(input) = 0`, so `output = input` exactly.

### 3. Use a low-rank bottleneck for any new attention module
Project D=1024 → d=256 inside the module. Keeps parameter count and memory small.

### 4. Separate learning rates
- Backbone (Qwen): lr=1e-4 or 1e-5 (slow, preserve pre-trained knowledge)
- New modules (temporal_emb, cross-attention): lr=1e-3 (fast, learning from scratch)

---

## Next Steps (priority order)

1. **Version B** — implement first. Easier (no Qwen internals), still a meaningful upgrade
   over simple addition. Test on Beauty with same settings as best Phase 3 sweep.

2. **Sentiment/Review embeddings** — add `reviewText` to the data pipeline and pre-compute
   review embeddings offline using the same Qwen backbone. Plug into Version B's
   cross-attention as a third structured signal alongside temporal and rating.

3. **Version A** — implement after Version B validates the direction. Requires hooking into
   Qwen's `embed_tokens` + `transformer_layers` separately. Higher implementation risk but
   potentially the strongest architecture.

4. **Ablation** — compare Version B (no review) vs Version B (with review) vs Version A
   on recall@10 to understand how much each addition actually helps.
