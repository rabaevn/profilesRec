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

Primary dev dataset: Amazon-2018 Beauty. Extended to Toys, Sports, Home,
MovieLens-1M, Steam, Yelp via a family-dispatch registry in `data.py`.

## Key reference numbers

- Beauty text-only backbone (all 4 enrich flags): **test R@10 = 0.0991**,
  val R@10 = 0.1056 (see `md_files/BEST_RUNS.md` for the full table).
- v6-CF fusion target on Beauty: beat val R@10 = 0.1017.
- Beauty June oracle head (r10_recovery, aux λ=0.3): val R@10 = 0.1012,
  ndcg@10 = 0.0565 (backup at `outputs/fusion_beauty_..._bak20260705/`).

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

## Current state (as of 2026-07-06)

Amazon oracle reruns **done and reproduce June** (val, filtered, gated oracle
r10 head): beauty R@10=0.1012 / ndcg@10=0.0565 (epoch-0 sanity = 0.1017
text-only baseline ✓), toys R@10=0.0882 / ndcg 0.0491, sports R@10=0.0316 /
ndcg 0.0158 (low but identical to June — sports is just a harder domain).
June backups at
`outputs/fusion_{beauty,toys,sports}_v6cf_gated_oracle_filtered_r10_bak20260705/`.

Steam: job 19085826 running under the 80G policy; resumes `train_fusion`
encoding from the emb cache (train queries fully cached, train profiles were
at shard 6/18; profile cache complete at 3,015,229 entries). Its predecessor
19063092 was cancelled by HPC IT for the 180G request.

**Next step once these finish: ablations.** The emb cache exists precisely so
ablation runs of `train_fusion.py` (same texts + same encoder) skip the
encoding pass — pass the same `--emb-cache-dir outputs/emb_cache_<tag>`.

Known gap: `slice_fusion.py` (pipeline step 6) does its own uncached encoding;
if Steam's step 6 becomes a bottleneck, wire it to the same cache.

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
