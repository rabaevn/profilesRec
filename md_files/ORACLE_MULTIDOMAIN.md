# Oracle Multi-Domain Pipeline

> **Status (2026-07-21): the oracle head is now an ABLATION, not the main
> method.** The thesis headline method is FiLM per-dimension gated fusion
> (no oracle aux) — see `md_files/FILM_FUSION_METHOD.md`. The oracle aux loss
> was found to *hurt* on Amazon and be unnecessary on Steam/ml1m
> (no-oracle ablation, 2026-07-16). This doc records the oracle-head results
> that FiLM is compared against; the shared pipeline steps 1-6 below still
> produce the backbone / profile cache / baseline slices that FiLM reuses.

Multi-domain replication of the Beauty oracle-R@10 pipeline
(`run_oracle_pipeline.sh`) across Toys, Sports, and Home_and_Kitchen.

## Pipeline (shared across all three domains)

Seven steps, identical config across domains, parameterised only by
`DATASET` and `TAG`:

1. **Backbone** — `train.py` with `Qwen/Qwen3-Embedding-0.6B`, `--enrich-text-input`, `--bf16` → `outputs/seqrec_qwen3_<tag>/best/`
2. **CF cooccur** — `build_cooccur.py` → `outputs/cooccur_<tag>.npz`
3. **CF context** — `build_cf_context.py` → `outputs/cf_context_<tag>.jsonl`
4. **CF user SVD** — `build_user_cf_emb.py` (k=64) → `outputs/user_cf_emb_<tag>.npz`
5. **v6 CoT+CF profiles** — `generate_profiles.py` → `outputs/profiles_<tag>_v6_cot_cf.jsonl`
6. **Filtered baseline slices** — `evaluate_slices.py --filter-seen-items --eval-catalog interacted`
7. **R@10-targeted oracle head** — `train_fusion.py --fusion-head-type gated_profile`
   - `gate_aux_objective=r10_recovery`, `gate_aux_lambda=0.3`, `gate_aux_alpha=0.3`, `gate_aux_pos_weight=balanced`
   - `gate_features=[history_len, has_profile, mean_history_pop, cos_q_p]`, `gate_mlp_hidden=16`, `gate_logit_init=-6.0`
   - `fusion_num_epochs=3`, `fusion_learning_rate=1e-3`, `fusion_batch_size=512`, `fusion_temperature=0.05`
8. **Slice fusion + analysis** — `slice_fusion.py` then `analyze_oracle_slices.py` (1000 bootstrap reps, popularity edges `[7, 12, 21, 40]`) → `outputs/fusion_<tag>_v6cf_gated_oracle_filtered_r10/slices/oracle_vs_baseline.json`

---

## Toys_and_Games (TAG=toys) — job 17995814 (DONE)

### Backbone (Qwen3-Embedding-0.6B, --enrich-text-input, bf16, 1 epoch)

| metric    | val (best) | test   |
|-----------|------------|--------|
| recall@5  | 0.0513     | 0.0482 |
| ndcg@5    | 0.0286     | 0.0271 |
| recall@10 | 0.0807     | 0.0757 |
| ndcg@10   | 0.0381     | 0.0359 |

Catalog: 11,865 items   Queries (val/test): 19,395 / 19,403

### Gated-profile oracle head (R@10 recovery)

Best at epoch 1 (gate still ~0; profile path barely engaged):

| metric    | val    |
|-----------|--------|
| recall@5  | 0.0616 |
| ndcg@5    | 0.0406 |
| recall@10 | 0.0882 |
| ndcg@10   | 0.0491 |

Epoch 0 sanity (zero-init proj): val recall@10 = 0.0900 (matches text-only baseline as expected).
Training trace: epoch 1 mean_gate=0.0083 → epoch 3 mean_gate=0.469, but ndcg@10 peaks at epoch 1.

### Slice analysis (filtered eval, 19,403 examples, 1000 bootstrap reps)

`oracle_prof` = gated head with profile features; `oracle_zero` = same head with profile features zeroed (counterfactual); `baseline` = text-only Qwen3.
Δ values are oracle_prof − baseline, percentage points (×100), with 95% bootstrap CIs.

| cohort                         | n      | base R@10 | o+p R@10 | o+0 R@10 | ΔR@10 vs base [95% CI]       | ΔNDCG@10 vs base [95% CI]     | mean gate |
|--------------------------------|--------|-----------|----------|----------|------------------------------|-------------------------------|-----------|
| FULL                           | 19,403 | 0.0835    | 0.0839   | 0.0856   | +0.04 [−0.18, +0.27]         | +0.41 [+0.30, +0.53]\*        | 0.0353    |
| cold (hist 3-5)                | 10,452 | 0.0907    | 0.0911   | 0.0932   | +0.04 [−0.26, +0.33]         | +0.46 [+0.30, +0.61]\*        | 0.0364    |
| long-hist (21+)                | 802    | 0.0623    | 0.0623   | 0.0623   | +0.00 [−1.12, +1.12]         | +0.36 [−0.10, +0.82]          | 0.0288    |
| popular (Q5)                   | 3,858  | 0.0645    | 0.0640   | 0.0664   | −0.05 [−0.44, +0.36]         | +0.27 [+0.07, +0.47]\*        | 0.0356    |
| long-tail (Q1)                 | 4,765  | 0.0888    | 0.0917   | 0.0907   | +0.29 [−0.13, +0.74]         | +0.63 [+0.40, +0.87]\*        | 0.0351    |
| cold × popular (3-5 × Q5)      | 2,124  | 0.0702    | 0.0687   | 0.0720   | −0.14 [−0.66, +0.38]         | +0.26 [−0.01, +0.53]          | 0.0365    |
| long-hist × long-tail (21+ × Q1)| 157   | 0.0382    | 0.0318   | 0.0382   | −0.64 [−3.18, +1.27]         | −0.02 [−0.83, +0.92]          | 0.0275    |
| cold × long-tail (3-5 × Q1)    | 2,633  | 0.0949    | 0.0980   | 0.0968   | +0.30 [−0.27, +0.87]         | +0.78 [+0.47, +1.12]\*        | 0.0363    |
| long-hist × popular (21+ × Q5) | 108    | 0.0648    | 0.0648   | 0.0648   | +0.00 [−3.70, +3.70]         | −0.04 [−1.20, +1.17]          | 0.0292    |
| has_profile=True (warm cache)  | 19,403 | 0.0835    | 0.0839   | 0.0856   | +0.04 [−0.19, +0.24]         | +0.41 [+0.30, +0.53]\*        | 0.0353    |
| has_profile=False (cold cache) | 0      | —         | —        | —        | —                            | —                             | —         |
| gate low (≤0.036, warm)        | 6,468  | 0.0889    | 0.0904   | 0.0912   | +0.15 [−0.25, +0.49]         | +0.50 [+0.31, +0.69]\*        | 0.0328    |
| gate mid (0.036-0.037, warm)   | 6,467  | 0.0872    | 0.0875   | 0.0891   | +0.03 [−0.34, +0.40]         | +0.45 [+0.26, +0.66]\*        | 0.0362    |
| gate high (>0.037, warm)       | 6,468  | 0.0744    | 0.0737   | 0.0765   | −0.06 [−0.42, +0.31]         | +0.29 [+0.11, +0.46]\*        | 0.0369    |

\* = 95% CI excludes zero.

**FULL transitions vs baseline**: gained=230, lost=222, kept=1,398, both_miss=17,553. McNemar b01/b10=230/222, p≈0.947 — no R@10 movement is statistically resolvable.

**Read across cohorts**: NDCG@10 lifts are reliably positive (re-ranking within the same hit set), but R@10 is flat. The strongest R@10-vs-baseline signal is cold × long-tail (+0.30pp, CI brushes zero). The oracle_zero column tracks oracle_prof closely, meaning the profile features carry almost none of the lift — most of what the head gains is from the residual path, not the LLM profile.

The has_profile=False cohort is empty because the v6_cot_cf cache was complete for the filtered eval set.

### Artifacts

- Backbone:        `outputs/seqrec_qwen3_toys/best/`
- CF cooccur:      `outputs/cooccur_toys.npz`
- CF context:      `outputs/cf_context_toys.jsonl`
- CF user SVD:     `outputs/user_cf_emb_toys.npz`
- Profile cache:   `outputs/profiles_toys_v6_cot_cf.jsonl`
- Baseline slices: `outputs/seqrec_qwen3_toys/slices_filtered/per_example.parquet`
- Oracle head:     `outputs/fusion_toys_v6cf_gated_oracle_filtered_r10/fusion_head.pt`
- Slice summary:   `outputs/fusion_toys_v6cf_gated_oracle_filtered_r10/slices/oracle_vs_baseline.json`
- SLURM logs:      `logs/oracle_toys_17995814.{out,err}`

---

## Sports_and_Outdoors (TAG=sports) — jobs 18018298 + 18026844 (IN FLIGHT)

### Backbone (Qwen3-Embedding-0.6B, --enrich-text-input, bf16, 1 epoch) — job 18018298 done

| metric    | val (best) | test   |
|-----------|------------|--------|
| recall@5  | 0.0157     | 0.0146 |
| ndcg@5    | 0.0090     | 0.0085 |
| recall@10 | 0.0294     | 0.0258 |
| ndcg@10   | 0.0134     | 0.0121 |

Catalog: 18,267 items   Queries (val/test): 35,597 / —   Train runtime: ~4h41m (16,864s, 47,075 steps).

### Gated-profile oracle head + slice analysis — pending

Job 18026844 is currently in step 5 (profile generation), 187,328 / 234,690 keys done (~80%) as of 2026-06-08. Awaiting steps 6–8 (filtered baseline slices → R@10 fusion head → slice analysis). Will be filled in once `outputs/fusion_sports_v6cf_gated_oracle_filtered_r10/slices/oracle_vs_baseline.json` exists.

---

## Home_and_Kitchen (TAG=home) — jobs 18018299 + 18026845 (IN FLIGHT)

### Backbone (Qwen3-Embedding-0.6B, --enrich-text-input, bf16, 1 epoch) — job 18018299 done

| metric    | val (best) | test     |
|-----------|------------|----------|
| recall@5  | 0.00102    | 0.00101  |
| ndcg@5    | 0.00069    | 0.00059  |
| recall@10 | 0.00194    | 0.00170  |
| ndcg@10   | 0.00098    | 0.00081  |

Catalog: 28,130 items   Queries (val/test): 66,519 / 66,519   Train runtime: ~7h58m (28,673s, 87,541 steps).

### Gated-profile oracle head + slice analysis — pending

Job 18026845 is currently in step 5 (profile generation), 118,336 / 433,962 keys done (~27%) as of 2026-06-08. Awaiting steps 6–8. Will be filled in once `outputs/fusion_home_v6cf_gated_oracle_filtered_r10/slices/oracle_vs_baseline.json` exists.
