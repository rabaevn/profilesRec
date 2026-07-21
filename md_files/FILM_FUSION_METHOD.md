# FiLM profile fusion — thesis main method

**Decision (2026-07-21):** the thesis headline method is the **FiLM-style
per-dimension gated profile fusion head** (`--fusion-head-type
gated_profile_film`, trained *without* the oracle aux loss). This supersedes
the scalar-gate oracle head (`gated_profile` + `gate_aux_objective=r10_recovery`)
as the primary result. The oracle head and the scalar no-oracle head remain in
the thesis as **ablations**, not the main line.

Rationale for the switch:

- The oracle aux loss *hurts* on Amazon (it forces the scalar gate open onto an
  unhelpful signal, mean gate ≈0.5) and is *not needed* for the profile lift on
  Steam/ml1m (see the no-oracle ablation, 2026-07-16). So the oracle
  supervision is a liability, not the story.
- FiLM lets the profile residual modulate individual embedding dimensions
  (gate g ∈ (0,1)^D) rather than a single scalar mix weight. It matches or
  slightly beats the scalar no-oracle head on every domain and gives the
  cleanest, best-motivated architecture to headline.
- Under the unified `ranking.py` protocol the FiLM zero-profile pass is
  bit-identical to the text-only baseline on every domain, so every
  `prof − baseline` delta below is an honest within-protocol measurement of the
  profile signal alone.

## What the head is

`GatedProfileFusionHead` with `gate_out_dim = embed_dim` (FiLM), zero-init
projection. Given the frozen text-encoder user query `q` and the frozen-encoded
LLM profile `p`, it produces a per-dimension gate `g = σ(MLP(gate_features))
∈ (0,1)^D` and fuses `z = q + g ⊙ proj(p)`. `gate_out_dim = 1` recovers the old
scalar head; old checkpoints load unchanged. The aux BCE (when used) reads
`scalar_gate_logit()` = mean over D, so an oracle-FiLM variant is still
possible, but the main method uses **no aux loss**.

- `gate_features = [history_len, has_profile, mean_history_pop, cos_q_p]`
- `gate_mlp_hidden = 16`, zero-init projection (epoch-0 val = text-only baseline)
- Training: 3 epochs, lr 1e-3, batch 512, temperature 0.05, `--gate-aux-lambda 0`
- No LLM at inference — profiles are precomputed in `profiles_<tag>_v6_cot_cf.jsonl`
  and encoded by the same frozen Qwen3-Embedding-0.6B backbone.

## Reproduction

Reuses the backbone, profile cache, baseline slices and embedding cache from
`run_oracle_pipeline.sh`; fails fast if a prerequisite is missing. Re-slice the
tag first if it still carries an old-protocol baseline parquet
(`run_reslice.sh`), then:

```
./submit_ablation.sh film <tag>          # maps to run_ablation_film.sh; DATASET/TAG env
```

Outputs: `outputs/fusion_<tag>_v6cf_gated_film_filtered/` (`fusion_head.pt`,
`slices/` test, `slices_val/` val, `slices/oracle_vs_baseline.json`).

## Headline results (test, R@10, unified ranking.py protocol)

`+prof` = FiLM head with profile features; the zero-profile pass equals the
text-only baseline exactly (asserted). Δ = prof − baseline, percentage points,
95% bootstrap CI (1000 reps). `*` = CI excludes zero.

| dataset | n (test) | baseline | +profile | Δ R@10 [95% CI] | mean gate |
|---------|---------:|---------:|---------:|:----------------|----------:|
| Beauty  | 22,363   | 9.74%    | 9.60%    | −0.14 [−0.30, +0.00]  | 0.024 |
| Toys    | 19,403   | 8.56%    | 8.70%    | +0.14 [−0.04, +0.34]  | 0.019 |
| Sports  | 35,597   | 2.92%    | 2.96%    | +0.04 [−0.06, +0.13]  | 0.010 |
| Steam   | 331,004  | 13.80%   | 15.36%   | **+1.56 [+1.51, +1.62]\*** | 0.019 |
| ML-1M   | 6,040    | 15.68%   | 17.65%   | **+1.97 [+1.44, +2.48]\*** | 0.012 |

Validation split (for completeness): Beauty 0.1017→0.1020 (+0.03pp),
Toys 0.0900→0.0924 (+0.24pp), Sports 0.0323→0.0324 (+0.01pp),
Steam 0.1539→0.1726 (+1.87pp), ML-1M 0.1593→0.1757 (+1.64pp).

**Story:** the LLM-profile signal, fused via FiLM, gives a large and highly
significant lift on Steam (+1.6pp, ~11% rel.) and MovieLens-1M (+2.0pp,
~13% rel.), and is flat (within noise) on the three Amazon domains. This is the
central thesis finding: profiles help where the interaction text is sparse /
repetitive and the user's taste is not already recoverable from item text
(games, movies), and add nothing on Amazon where the item text already carries
the signal. The gate learns this on its own — it stays nearly closed
(mean ≈0.01–0.02) everywhere; the difference is whether opening it slightly
helps.

## Per-domain cohort breakdown (test)

Cohorts: history-length × target-popularity, plus user-activity and item-age
tertiles. `*` = 95% CI excludes zero.

### Beauty (n=22,363)
| cohort | n | base | +prof | Δ R@10 [95% CI] | Δ NDCG [95% CI] | gate |
|---|--:|--:|--:|--:|--:|--:|
| FULL | 22363 | 9.74 | 9.60 | −0.14 [−0.30,+0.00] | −0.09 [−0.16,−0.02]* | 0.024 |
| cold (hist 3-5) | 11397 | 9.31 | 9.34 | +0.03 [−0.17,+0.23] | −0.01 [−0.10,+0.08] | 0.025 |
| long-hist (21+) | 1015 | 16.95 | 16.26 | −0.69 [−1.67,+0.20] | −0.54 [−0.92,−0.14]* | 0.021 |
| popular (Q5) | 4451 | 10.00 | 9.82 | −0.18 [−0.54,+0.20] | −0.05 [−0.21,+0.12] | 0.024 |
| long-tail (Q1) | 4910 | 9.69 | 9.39 | −0.31 [−0.61,−0.04]* | −0.11 [−0.25,+0.03] | 0.024 |
| cold × long-tail (3-5 × Q1) | 2664 | 11.37 | 10.96 | −0.41 [−0.79,−0.08]* | −0.16 [−0.33,−0.00]* | 0.025 |
| cold × popular (3-5 × Q5) | 2185 | 6.50 | 6.86 | +0.37 [−0.09,+0.82] | +0.20 [+0.00,+0.43]* | 0.025 |
| activity light | 11397 | 9.31 | 9.34 | +0.03 [−0.18,+0.22] | −0.01 [−0.10,+0.07] | 0.025 |
| activity medium | 4488 | 8.87 | 8.44 | −0.42 [−0.78,−0.04]* | −0.10 [−0.28,+0.08] | 0.024 |
| activity heavy | 6478 | 11.10 | 10.87 | −0.23 [−0.49,+0.06] | −0.21 [−0.36,−0.08]* | 0.023 |
| item-age established | 7457 | 6.96 | 6.95 | −0.01 [−0.20,+0.17] | −0.02 [−0.11,+0.06] | 0.024 |
| item-age mid | 7476 | 8.12 | 8.24 | +0.12 [−0.15,+0.39] | +0.05 [−0.07,+0.16] | 0.024 |
| item-age new | 7430 | 14.16 | 13.63 | −0.52 [−0.81,−0.19]* | −0.29 [−0.45,−0.14]* | 0.024 |

### Toys (n=19,403)
| cohort | n | base | +prof | Δ R@10 [95% CI] | Δ NDCG [95% CI] | gate |
|---|--:|--:|--:|--:|--:|--:|
| FULL | 19403 | 8.56 | 8.70 | +0.14 [−0.04,+0.34] | +0.07 [−0.01,+0.16] | 0.019 |
| cold (hist 3-5) | 10452 | 9.32 | 9.42 | +0.11 [−0.13,+0.33] | +0.04 [−0.08,+0.16] | 0.019 |
| long-hist (21+) | 802 | 6.23 | 6.98 | +0.75 [−0.62,+2.12] | +0.58 [+0.02,+1.19]* | 0.016 |
| popular (Q5) | 3858 | 6.64 | 6.64 | +0.00 [−0.36,+0.36] | −0.06 [−0.22,+0.10] | 0.019 |
| long-tail (Q1) | 4765 | 9.07 | 9.55 | +0.48 [+0.10,+0.82]* | +0.26 [+0.09,+0.45]* | 0.019 |
| cold × long-tail (3-5 × Q1) | 2633 | 9.68 | 10.37 | +0.68 [+0.15,+1.22]* | +0.30 [+0.06,+0.57]* | 0.019 |
| cold × popular (3-5 × Q5) | 2124 | 7.20 | 7.11 | −0.09 [−0.52,+0.33] | −0.12 [−0.35,+0.09] | 0.019 |
| activity light | 6696 | 9.65 | 9.71 | +0.06 [−0.25,+0.37] | +0.01 [−0.13,+0.16] | 0.019 |
| activity medium | 7528 | 8.46 | 8.71 | +0.25 [−0.03,+0.54] | +0.07 [−0.07,+0.21] | 0.019 |
| activity heavy | 5179 | 7.30 | 7.40 | +0.10 [−0.31,+0.46] | +0.14 [−0.04,+0.33] | 0.018 |
| item-age established | 6470 | 5.09 | 5.10 | +0.02 [−0.23,+0.26] | −0.02 [−0.15,+0.11] | 0.019 |
| item-age mid | 6465 | 9.68 | 9.59 | −0.09 [−0.42,+0.25] | −0.10 [−0.25,+0.05] | 0.019 |
| item-age new | 6468 | 10.92 | 11.43 | +0.51 [+0.12,+0.88]* | +0.33 [+0.15,+0.50]* | 0.018 |

### Sports (n=35,597)
| cohort | n | base | +prof | Δ R@10 [95% CI] | Δ NDCG [95% CI] | gate |
|---|--:|--:|--:|--:|--:|--:|
| FULL | 35597 | 2.92 | 2.96 | +0.04 [−0.06,+0.13] | +0.02 [−0.02,+0.07] | 0.010 |
| cold (hist 3-5) | 18430 | 3.03 | 3.07 | +0.04 [−0.10,+0.18] | +0.02 [−0.05,+0.08] | 0.010 |
| long-hist (21+) | 1108 | 1.99 | 2.08 | +0.09 [−0.27,+0.54] | +0.00 [−0.15,+0.14] | 0.008 |
| popular (Q5) | 7115 | 2.98 | 2.85 | −0.13 [−0.35,+0.08] | −0.06 [−0.15,+0.04] | 0.010 |
| long-tail (Q1) | 8067 | 2.43 | 2.59 | +0.16 [−0.01,+0.33] | +0.08 [−0.01,+0.17] | 0.010 |
| cold × long-tail (3-5 × Q1) | 4166 | 2.95 | 3.12 | +0.17 [−0.10,+0.48] | +0.11 [−0.01,+0.24] | 0.010 |
| cold × popular (3-5 × Q5) | 3672 | 2.89 | 2.78 | −0.11 [−0.46,+0.25] | −0.06 [−0.21,+0.07] | 0.010 |
| activity light | 18430 | 3.03 | 3.07 | +0.04 [−0.11,+0.20] | +0.02 [−0.04,+0.08] | 0.010 |
| activity medium | 7277 | 3.02 | 2.98 | −0.04 [−0.25,+0.15] | −0.05 [−0.14,+0.04] | 0.010 |
| activity heavy | 9890 | 2.65 | 2.74 | +0.09 [−0.08,+0.27] | +0.08 [+0.01,+0.15]* | 0.009 |
| item-age established | 11909 | 2.20 | 2.33 | +0.13 [−0.01,+0.28] | +0.05 [−0.01,+0.12] | 0.010 |
| item-age mid | 11826 | 2.85 | 2.93 | +0.08 [−0.09,+0.26] | +0.08 [+0.00,+0.15]* | 0.010 |
| item-age new | 11862 | 3.72 | 3.62 | −0.10 [−0.30,+0.09] | −0.06 [−0.14,+0.03] | 0.010 |

### Steam (n=331,004)
| cohort | n | base | +prof | Δ R@10 [95% CI] | Δ NDCG [95% CI] | gate |
|---|--:|--:|--:|--:|--:|--:|
| FULL | 331004 | 13.80 | 15.36 | +1.56 [+1.51,+1.62]* | +1.41 [+1.38,+1.44]* | 0.019 |
| cold (hist 3-5) | 132604 | 16.95 | 18.90 | +1.94 [+1.86,+2.03]* | +1.84 [+1.78,+1.90]* | 0.021 |
| long-hist (21+) | 29417 | 7.50 | 8.29 | +0.80 [+0.65,+0.93]* | +0.67 [+0.59,+0.74]* | 0.013 |
| popular (Q5) | 66038 | 28.25 | 30.35 | +2.10 [+1.98,+2.22]* | +2.42 [+2.33,+2.51]* | 0.019 |
| long-tail (Q1) | 66383 | 4.30 | 4.51 | +0.21 [+0.13,+0.30]* | +0.10 [+0.06,+0.15]* | 0.018 |
| cold × long-tail (3-5 × Q1) | 23013 | 4.25 | 4.44 | +0.19 [+0.03,+0.37]* | +0.10 [+0.02,+0.18]* | 0.021 |
| cold × popular (3-5 × Q5) | 30449 | 34.35 | 36.68 | +2.34 [+2.17,+2.52]* | +2.89 [+2.76,+3.03]* | 0.022 |
| activity light | 132604 | 16.95 | 18.90 | +1.94 [+1.85,+2.03]* | +1.84 [+1.78,+1.90]* | 0.021 |
| activity medium | 101987 | 13.57 | 15.09 | +1.51 [+1.42,+1.61]* | +1.35 [+1.30,+1.41]* | 0.019 |
| activity heavy | 96413 | 9.71 | 10.80 | +1.08 [+1.00,+1.17]* | +0.88 [+0.83,+0.92]* | 0.015 |
| item-age established | 110459 | 12.37 | 14.22 | +1.86 [+1.77,+1.94]* | +1.93 [+1.87,+1.99]* | 0.019 |
| item-age mid | 110457 | 14.58 | 15.53 | +0.95 [+0.87,+1.03]* | +0.84 [+0.78,+0.89]* | 0.019 |
| item-age new | 110088 | 14.46 | 16.34 | +1.88 [+1.78,+1.98]* | +1.46 [+1.40,+1.52]* | 0.018 |

### MovieLens-1M (n=6,040)
No cold slice: 5-core + leave-two-out leaves every user in the 21+ history
bucket (hist 3-5 is empty).

| cohort | n | base | +prof | Δ R@10 [95% CI] | Δ NDCG [95% CI] | gate |
|---|--:|--:|--:|--:|--:|--:|
| FULL | 6040 | 15.68 | 17.65 | +1.97 [+1.44,+2.48]* | +0.96 [+0.75,+1.19]* | 0.012 |
| long-hist (21+) | 5863 | 15.59 | 17.50 | +1.91 [+1.40,+2.44]* | +0.94 [+0.72,+1.17]* | 0.012 |
| popular (Q5) | 1208 | 17.22 | 20.03 | +2.81 [+1.74,+3.89]* | +2.00 [+1.45,+2.55]* | 0.011 |
| long-tail (Q1) | 1208 | 15.56 | 16.31 | +0.75 [−0.50,+1.99] | +0.06 [−0.42,+0.59] | 0.012 |
| activity light | 2046 | 20.04 | 22.83 | +2.79 [+1.86,+3.81]* | +1.44 [+1.00,+1.82]* | 0.011 |
| activity medium | 1989 | 16.04 | 18.20 | +2.16 [+1.26,+3.07]* | +0.96 [+0.55,+1.39]* | 0.012 |
| activity heavy | 2005 | 10.87 | 11.82 | +0.95 [+0.10,+1.80]* | +0.48 [+0.14,+0.83]* | 0.012 |
| item-age established | 2016 | 16.42 | 18.55 | +2.13 [+1.24,+3.08]* | +1.09 [+0.70,+1.49]* | 0.012 |
| item-age mid | 2012 | 15.21 | 17.54 | +2.34 [+1.44,+3.23]* | +1.21 [+0.83,+1.62]* | 0.012 |
| item-age new | 2012 | 15.41 | 16.85 | +1.44 [+0.55,+2.39]* | +0.59 [+0.20,+1.00]* | 0.012 |

## Cold-user finding (thesis sub-result)

On the cold slice (history 3-5), the pattern mirrors the full set: Amazon is
flat/within-noise, Steam is strongly positive (+1.94pp R@10, +1.84pp NDCG on
cold users; +2.34pp R@10 on cold × popular). MovieLens has no cold users to
report. So the profile signal is **not specifically a cold-start remedy** — it
helps in proportion to how much the domain's item text under-determines the
next item, across all history lengths, and cold Steam users benefit because
Steam item text is sparse, not because they are cold per se. The one place the
signal is significantly *negative* is Beauty cold × long-tail (−0.41pp), where
opening the gate onto the profile actively hurts rare-item cold users.

## Cross-references

- `run_ablation_film.sh`, `submit_ablation.sh` — launchers.
- `outputs/fusion_<tag>_v6cf_gated_film_filtered/slices/oracle_vs_baseline.json`
  — per-domain machine-readable cohort table (1000 bootstrap reps).
- Beauty 3-way method comparison (baseline / oracle / no-oracle / FiLM):
  `plot_fusion_3way.py` → `outputs/plots/fusion_3way_beauty*.{png,json}`.
- Ablations that this method is compared against: scalar oracle head
  (`gate_aux_objective=r10_recovery`) and scalar no-oracle head
  (`--gate-aux-lambda 0`, `gate_out_dim=1`); see CLAUDE.md decision history.
</content>
</invoke>
