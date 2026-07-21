# CLAUDE.md — project memory for seqrec

This file is Claude's cross-session memory for this project. It is loaded
automatically in any Claude session with repo access (web, CLI, IDE). Keep it
current: when a milestone lands or a decision changes, update the relevant
section and the "Current state" date.

## What this project is

M.Sc. thesis (Neomi Rabaev): sequential recommendation with a two-tower
text-encoder backbone (Qwen3-Embedding-0.6B via sentence-transformers) plus an
**LLM-distilled user-preference signal** fused at the user tower. Advisor
direction: "encode user preferences also". Hard constraint: **no LLM calls at
inference** — LLM-generated user profiles are precomputed into a JSONL cache
and encoded by the same frozen text encoder.

**Main method (thesis headline, decided 2026-07-21):** FiLM-style
per-dimension gated profile fusion — `--fusion-head-type gated_profile_film`,
trained *without* the oracle aux loss. The scalar oracle head and scalar
no-oracle head are ablations, not the main line. Full write-up +
5-dataset results in `md_files/FILM_FUSION_METHOD.md`.

Primary dev dataset: Amazon-2018 Beauty. Extended to Toys, Sports, Home,
MovieLens-1M, Steam, Yelp via a family-dispatch registry in `data.py`.

## Key reference numbers

- Beauty text-only backbone (all 4 enrich flags): **test R@10 = 0.0991**,
  val R@10 = 0.1056 (see `md_files/BEST_RUNS.md` for the full table).
- v6-CF fusion target on Beauty: beat val R@10 = 0.1017.
- Beauty June oracle head (r10_recovery, aux λ=0.3): val R@10 = 0.1012,
  ndcg@10 = 0.0565 (backup at `outputs/fusion_beauty_..._bak20260705/`).
- Beauty filtered slice-protocol baseline (unified `ranking.py`, 2026-07-19):
  **test R@10 = 0.0974**, val R@10 = 0.1017 (the old 0.0949 test number was a
  gold-score off-by-one artifact — see 2026-07-19 decision entry).
- **FiLM main-method headline (test R@10, prof vs baseline, 2026-07-21):**
  Steam 0.1380 → **0.1536 (+1.56pp\*, ~11% rel.)**, ml1m 0.1568 → **0.1765
  (+1.97pp\*, ~13% rel.)**; Amazon flat within noise — Beauty 0.0974→0.0960
  (−0.14pp), toys 0.0856→0.0870 (+0.14pp), sports 0.0292→0.0296 (+0.04pp).
  `*`=CI excludes 0. Full tables: `md_files/FILM_FUSION_METHOD.md`.

## Decision history (condensed)

- **2026-05-12, F3 result:** soft-prompt suffix fusion (F3) **tied** with the
  λ=0 baseline across 3 seeds on Beauty — raw LLM-profile signal lift was
  below the noise floor. Do not revisit soft-prompt fusion.
- **2026-05-17, pivot:** to v5-DPO + v5-prompt profile cache and the
  Qwen3-Embedding-0.6B backbone.
- **2026-05-26, fallback plan:** if the rerank track can't beat text-encoder
  R@10=0.094, pivot to collaborative signals + retrieval-CoT → became the
  **v6 CoT+CF profile track** (`profiles_<tag>_v6_cot_cf.jsonl`), which is the
  current line of work.
- **2026-05-28, CF fusion gotchas:** cohort filter bug, unbounded α, frozen
  SVD. Fix pattern: learnable residual gate + `has_cf` mask + zero-init
  projection (gate starts "off", epoch-0 eval must equal text-only baseline).
- **2026-06-04, multi-domain oracle:** oracle-R@10 pipeline parameterized by
  `$DATASET`/`$TAG` (`run_oracle_pipeline.sh` + `submit_oracle_domain.sh`).
  Motivation: Beauty's fusion lift concentrates in the dense `hl=21+ × Q2/Q3`
  cohort; thesis must show it generalizes across domains.
- **2026-06-26, cohort axes:** added user-activity and item-age axes
  (`cohort_dims.py`); newer items benefit most from the profile signal.
- **2026-06-27, Steam/Yelp context overflow:** profile prompts exceed 4096
  tokens; fix = char caps in `build_profile_prompt` + `REVIEW_WORDS=8`.
  **Never change `max_history` — it defines profile-cache keys.**
- **2026-07-05, Steam OOM + emb cache:** Steam oracle job 18856386 OOM-killed
  (94G peak vs 90G) 18h into `train_fusion`'s encoding pass. Added
  `train_fusion.py --emb-cache-dir`: sharded, resumable, content-fingerprinted
  (+ encoder path+mtime salt) embedding cache at `outputs/emb_cache_<tag>/`.
  Launcher sets `PYTHONUNBUFFERED=1`.
- **2026-07-06, cluster policy + memory diet:** HPC IT cancelled the 180G
  Steam job — **max 80G RAM with an rtx_6000, never set cpus-per-task**.
  `train_fusion.py` was slimmed to fit: profile cache dict freed after
  resolution, `history_text` strings freed after query encoding, raw
  interactions freed after `item_log_pop`, and train targets stored as
  catalog *indices* gathered on-GPU per batch instead of a materialized
  n×1024 float32 array (~12G on Steam).
- **2026-07-09, repeat-target eval discrepancy (Steam):** 12.8% of Steam test
  examples have the gold target already in the user's history (repeat
  interactions; ~0 in Amazon 5-core). `evaluate_slices.py` scores the gold
  *before* masking seen items (gold always rankable → 81% hit on those rows),
  while `train_fusion`/`slice_fusion` mask all seen items incl. the gold
  (guaranteed miss). Hence baseline R@10 0.137 vs fusion-side 0.034 — an
  artifact, not a regression. **RESOLVED 2026-07-19 (user decision):** the
  convention is now *gold always rankable* — `ranking.py` gathers the gold
  score before masking seen items, so repeat targets can be hits and the
  full test set is meaningful. Applies uniformly to baseline and all heads;
  bit-identical on domains with 0 repeats (all Amazon tags, ml1m). Old
  Steam subset-based numbers (`plot_steam_oracle.py`) predate this.
- **2026-07-15, cross-script eval gap (all domains):** `slice_fusion`'s
  zero-profile pass forces the gate off, making the head mathematically the
  identity (= text-only baseline), yet its metrics beat `evaluate_slices`'
  baseline everywhere (Beauty test: R@10 0.0974 vs 0.0949, NDCG 0.0544 vs
  0.0486; ranks differ on 41% of rows, median shift 1 position). Any
  fusion-vs-baseline delta that mixes the two scripts is contaminated by
  protocol noise of the same magnitude as the claimed Amazon effects. The
  honest within-protocol comparison is **prof vs zero**: on the Amazon test
  sets it is ~0 or slightly negative (Beauty −0.26pp R@10), while Steam
  (+0.60pp R@10) and ml1m (+2.2pp R@10) show genuine profile lift. Also:
  `make_preliminary_plots.py`'s headline plot labels `oracle_zero` as
  "Oracle-gated profile" (it is the zero-profile identity pass) and its
  suptitle says "validation" but the data is test/filter-seen. Full-test
  3-bar comparison: `plot_oracle_full_test.py` →
  `outputs/plots/oracle_full_test.png`.
- **2026-07-19, protocol gap ROOT-CAUSED + FIXED:** `evaluate_slices.py`
  computed the gold score via a numpy elementwise path while catalog scores
  used a BLAS matmul; float32 rounding differences made the gold count
  *against itself* → rank inflated by exactly +1 on ~41% of rows.
  `slice_fusion`'s numbers were the correct ones; the old baselines were the
  artifact (Beauty test R@10 is 0.0974, not 0.0949). Fix: all ranking goes
  through `ranking.py:per_example_ranks` (gold gathered from the same score
  matrix, TF32 off, gold always rankable even if seen); both eval scripts share the
  train-time `--emb-cache-dir` and grew a `--split {test,val}` flag; the
  zero-profile pass ranks the raw cached query (an extra `normalize(q)` on
  unit vectors shifts values ~1e-8 and flipped near-ties between duplicate
  item texts). After `run_reslice.sh` (re-slice only, no retraining), the
  zero pass of every head is **bit-identical** to the baseline on Beauty
  val+test. Old-protocol slices backed up at `*_bak20260719`. Amazon
  toys/sports (and Steam/ml1m) still carry old-protocol baseline parquets —
  rerun `run_reslice.sh` per tag before quoting cross-script deltas.
- **2026-07-21, FiLM promoted to thesis main method:** the FiLM per-dimension
  gated head (`gated_profile_film`, no oracle aux) is now the headline method;
  the scalar oracle head and scalar no-oracle head are demoted to ablations.
  Reasons: oracle aux *hurts* on Amazon and is unnecessary on Steam/ml1m
  (2026-07-16 no-oracle ablation), and FiLM's per-dimension gate is the
  best-motivated architecture and matches/beats the scalar heads everywhere.
  All 5 domains run under the unified protocol (zero pass = baseline,
  asserted): Steam +1.56pp\* and ml1m +1.97pp\* R@10 (both CI-clear), Amazon
  flat within noise. Full write-up + per-domain cohort tables in
  `md_files/FILM_FUSION_METHOD.md`.

## Current state (as of 2026-07-21)

**FiLM is the thesis main method (decided 2026-07-21).** All five domains done
under the unified `ranking.py` protocol via `./submit_ablation.sh film <tag>`
(`run_ablation_film.sh`, `--fusion-head-type gated_profile_film`,
`--gate-aux-lambda 0`; outputs `outputs/fusion_<tag>_v6cf_gated_film_filtered/`).
Every head's zero-profile pass is bit-identical to the text-only baseline
(asserted), so every delta below is an honest within-protocol profile measurement.

Headline (test R@10, prof vs baseline):
- Steam 0.1380 → **0.1536 (+1.56pp\*, ~11% rel.)** — job 19451347.
- ml1m 0.1568 → **0.1765 (+1.97pp\*, ~13% rel.)** — job 19450474.
- Beauty 0.0974 → 0.0960 (−0.14pp), toys 0.0856 → 0.0870 (+0.14pp),
  sports 0.0292 → 0.0296 (+0.04pp) — all Amazon flat within noise
  (jobs 19449410 / 19450468 / 19450470).

`*` = 95% bootstrap CI excludes 0. Mean gate stays nearly closed everywhere
(0.01–0.02); the profile only helps where item text under-determines the next
item (games, movies), not on Amazon where item text already carries the signal.
Cold-slice sub-result (hist 3-5): same pattern — Steam +1.94pp\* on cold users,
Amazon flat, ml1m has no cold slice (5-core leaves all users long-history).
Full per-domain cohort tables: `md_files/FILM_FUSION_METHOD.md`.

**Next:** headline plots/tables for the thesis writeup; a non-Amazon dataset
with genuine cold users (Goodreads / MIND) to test whether Steam's cold-start
lift reproduces where the domain is text-rich but cold.

## Previous state (as of 2026-07-19, superseded by FiLM promotion above)

**Ablation 2 (FiLM per-dimension gate) DONE on Beauty 2026-07-19** (job
19449410; `run_ablation_film.sh`, `--fusion-head-type gated_profile_film`,
no oracle aux). `GatedProfileFusionHead` now takes `gate_out_dim` (1 = scalar,
embed_dim = FiLM g∈(0,1)^D); old checkpoints load unchanged; aux BCE uses
`scalar_gate_logit()` (mean over D) so an oracle-FiLM run stays possible.
Epoch-0 sanity 0.1017 ✓. **Beauty 3-way comparison under the unified
protocol** (`plot_fusion_3way.py` → `outputs/plots/fusion_3way_beauty*.{png,json}`;
zero pass of all heads bit-identical to baseline, asserted):
test baseline R@10 0.0974 — oracle prof 0.0948 (−0.26pp, CI excl. 0),
no-oracle 0.0964 (−0.10pp), FiLM 0.0960 (−0.14pp); val baseline 0.1017 —
oracle 0.1012, no-oracle 0.1015, FiLM 0.1020 (+0.03pp, CI [−0.14,+0.20]).
FiLM mean gate 0.024. Takeaway: on Beauty all three fusion methods are ≈0
vs baseline; the story stays "profiles help on Steam/ml1m, not Amazon" —
FiLM on Steam/ml1m is the interesting next run
(`./submit_ablation.sh film steam ml1m` after re-slicing those tags).

## Previous state (as of 2026-07-09)

Amazon oracle reruns **done and reproduce June** (val, filtered, gated oracle
r10 head): beauty R@10=0.1012 / ndcg@10=0.0565 (epoch-0 sanity = 0.1017
text-only baseline ✓), toys R@10=0.0882 / ndcg 0.0491, sports R@10=0.0316 /
ndcg 0.0158 (low but identical to June — sports is just a harder domain).
June backups at
`outputs/fusion_{beauty,toys,sports}_v6cf_gated_oracle_filtered_r10_bak20260705/`.

**Steam DONE** (job 19085826, 6h29m, peak 77.6G — emb cache resume worked).
Protocol-corrected results (see gotcha below; plot at
`outputs/plots/steam_oracle.png`, script `plot_steam_oracle.py`), on the
target∉history subset (n=288,535): backbone R@10=0.0381 → fusion+profile
0.0461 (+0.8pp, ~21% rel), fusion+zero-profile 0.0392. Profile lift is
positive across nearly all hl×pop cohorts.

**MovieLens-1M DONE** (job 19270872): test baseline R@10=0.1523 →
fusion+profile **0.179** (+2.7pp, ~17.5% rel; prof−zero CI [+1.6,+2.8]pp),
zero-profile 0.1568. No repeat-target issue (one rating per movie), so
`oracle_vs_baseline.json` is valid as-is for ml1m.

Yelp: **dropped** (2026-07-09, user decision — raw data is license-gated).
If revived: place `yelp_academic_dataset_{business,review}.json[.gz]` under
`raw_data/yelp/`, `./submit_oracle_domain.sh yelp`, and apply the
repeat-target protocol treatment (repeat restaurant visits).

Amazon code-alignment reruns submitted 2026-07-09 (beauty=19274169,
toys=19274170, sports=19274171): same pipeline but under the post-memory-diet
`train_fusion.py`, so every domain's result comes from the same code version.
Expected fast (emb caches warm). Jul-5 results backed up at
`*_bak20260709/`; expect identical numbers.

**Ablation 1 (no-oracle) DONE 2026-07-16** (jobs 19384312–15): same gated
fusion head but `--gate-aux-lambda 0` — no oracle labels / aux BCE, gate
learns from the main loss only. Scripts: `run_ablation_no_oracle.sh` via the
generic `./submit_ablation.sh <name> <tags...>` (maps name →
`run_ablation_<name>.sh`; fails fast if prereqs missing). Outputs:
`outputs/fusion_<tag>_v6cf_gated_nooracle_filtered/`.
**Result (prof−zero R@10, slice_fusion protocol; Steam on target∉history):**
oracle vs no-oracle — beauty −0.26pp vs −0.10pp, toys −0.16pp vs −0.02pp,
sports −0.15pp (CI excl. 0) vs +0.05pp, steam +0.69pp vs **+0.73pp**.
Takeaway: oracle supervision is not needed for Steam's profile lift and
*hurts* on Amazon — it forces the gate open (mean gate ≈0.5) onto an unhelpful
signal, while the unsupervised gate stays nearly closed (0.01–0.04) and does
no harm. The zero columns are identical across heads by construction.

**Next steps: more ablations** (reuse `--emb-cache-dir outputs/emb_cache_<tag>`),
+ non-Amazon dataset
candidates beyond ml1m/steam/yelp: Goodreads (UCSD Book Graph) and MIND
(Microsoft News) fit best — both have rich item text; needs a new family
loader in `data.py`.

~~Known gap: `slice_fusion.py` (pipeline step 6) does its own uncached
encoding~~ — closed 2026-07-19: both `evaluate_slices.py` and
`slice_fusion.py` take `--emb-cache-dir` (wired in the pipeline/ablation
shell scripts), so Steam re-slices are cheap.

## Conventions and gotchas

- Per-domain artifacts: `outputs/seqrec_qwen3_<tag>/`, `cooccur_<tag>.npz`,
  `cf_context_<tag>.jsonl`, `user_cf_emb_<tag>.npz`,
  `profiles_<tag>_v6_cot_cf.jsonl`,
  `fusion_<tag>_v6cf_gated_oracle_filtered_r10/`.
- The pipeline is idempotent via existence gates; to force a rerun of a step,
  rename/remove its output (back up, don't delete — June results are the
  cross-check baseline).
- `--enrich-text-input` = sep + time + rating + pos-marker history flags;
  synergistic, always on for backbones and fusion.
- Gated fusion heads: zero-init projection means epoch-0 val must equal the
  text-only baseline — if it doesn't, something is wrong.
- Steam/Yelp: `ENCODE_BS=8` (long histories, 24GB GPU), `REVIEW_WORDS=8`.
- **Cluster policy (HPC IT):** max 80G RAM with an rtx_6000 GPU; never set
  `--cpus-per-task` (per-GPU default applies); no GPU for CPU-only jobs.
- After training, **push checkpoints to Hugging Face Hub** — local disk quota
  is tight; don't leave large checkpoints on disk.
- SLURM logs: `logs/<jobname>_<jobid>.{out,err}`. Check OOM via
  `sacct -j <id> --format=JobID,State,MaxRSS,ReqMem`.

## Where the details live

- `md_files/BEST_RUNS.md` — best runs + hyperparameters per phase.
- `md_files/ORACLE_MULTIDOMAIN.md` — per-domain oracle pipeline results.
- `md_files/MEETING_NEXT_STEPS.md` — advisor meeting notes and phased plan.
- `md_files/DOCUMENTATION.md`, `ARCHITECTURE_IDEAS.md`,
  `IMPROVEMENT_IDEAS.md` — background and ideas.
