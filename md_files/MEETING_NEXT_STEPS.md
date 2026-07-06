# Meeting Summary & Next Steps

Building on the LR-scaled batch sweep baseline (BS=256, NDCG@10 ≈ 0.0360, Recall@10 ≈ 0.0760 on Amazon Beauty 2018).

## Meeting Summary

**Goal:** improve sequence-recommendation quality on top of the BS=256 / LR-scaled baseline.

**Directions discussed**

1. **Stronger encoder** — swap MiniLM-L6 for a bigger backbone.
2. **Temporal modulation**
   - (a) Text-level: inject relative time ("3 days ago", "1 week ago") into the history string.
   - (b) Embedding-level: add a separate temporal embedding (e.g. learned bucketed-Δt vector summed/concatenated with the item embedding).
3. **Ranking / sentiment signal**
   - Text-level: include rating (e.g. "★4") or a sentiment token in the history.
   - Embedding-level: weight items by rating/sentiment, or add a side embedding.
4. **Noise filtering** — acknowledged hard, deferred.
5. **Multi-interest / multi-vector** — represent a user with several vectors instead of one; requires architectural change (e.g. MIND-style multi-interest extraction or `Asym` two-tower with multiple query heads).
6. **Open research question:** what is the optimal way to embed a user history for seq-rec?

## Suggested Next Steps (by impact-to-effort)

Phase it so cheap wins land first and become the new baseline before architectural changes.

### Phase 1 — lock in a stronger baseline (config-only)

Re-run the LR-scaled BS=256 recipe with:
- `BAAI/bge-base-en-v1.5`
- `intfloat/e5-base-v2`
- `sentence-transformers/all-mpnet-base-v2`

Pick the winner; this becomes the reference for everything else.

### Phase 2 — text-level ablations (single grid, all comparable)

Controlled ablation over the Phase 1 winner. All are small edits to `make_history_text`:

- `+time_text`: append "(N days ago)" per item.
- `+rating_text`: prepend "★R" per item.
- `+sep`: explicit `[SEP]` between items (independent control variable, recommended regardless).
- `+pos_marker`: ordinal markers `[1]…[N]`.

A 2×2 (time × rating) plus the `+sep` baseline gives 5 runs and answers "does any of this help at all" cheaply.

### Phase 3 — embedding-level (architectural)

Only if Phase 2 shows time/rating actually carries useful signal:

- **Temporal embedding:** bucket Δt into log-spaced bins, learn an embedding table, fuse with item vectors before pooling.
- **Rating-weighted pooling:** scale per-item contribution by rating before mean-pool.

These need code changes in the encoding path; worth doing only after the text-level ablation justifies it.

### Phase 4 — multi-interest / multi-vector

Largest scope. Two reasonable framings:

- **Multi-vector query:** K learned attention heads over history → K query vectors; score = max over heads. Closest to MIND.
- **Two-tower with capsule/cluster head:** cluster history items, embed each cluster separately.

Defer until Phase 1–3 plateau, since these change evaluation (max-over-heads) and complicate ablation.

### Deferred

- Noise removal (agreed).
- Hard-negative mining is missing from the meeting list but is probably higher ROI than multi-interest — worth flagging.

## On the broader question

"How to optimally embed history for seq-rec" is really three sub-questions that the phasing above separates:

1. What **content** goes in (text fields, time, rating).
2. What **form** it takes (text vs. side embedding).
3. Whether one vector is enough (single vs. multi-interest).

Treating them in that order keeps the experiments interpretable.
