#!/bin/bash
#SBATCH --job-name=abl_film
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=80G
#SBATCH --time=6-23:00:00
# Cluster policy (HPC IT, 2026-07-06): max 80G RAM with an rtx_6000 GPU, and
# do NOT set cpus-per-task — the per-GPU default applies.

# Ablation: FiLM-style per-dimension gate (--fusion-head-type
# gated_profile_film) — gate vector g in (0,1)^D instead of a scalar, so the
# profile residual can modulate individual embedding dimensions. Trained
# WITHOUT the oracle aux loss (--gate-aux-lambda 0), matching the no-oracle
# ablation finding that oracle supervision is not needed for profile lift.
# Reuses the backbone, profile cache, baseline slices and emb cache produced
# by run_oracle_pipeline.sh — fails fast if any prerequisite is missing.
#
# Required env vars:
#   DATASET  dataset name, must exist in data.DATASET_REGISTRY
#   TAG      short slug used in output paths (e.g. beauty, toys, sports, steam)

source ~/.bashrc
conda activate my_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

set -euo pipefail

: "${DATASET:?Set DATASET env var (e.g. Beauty)}"
: "${TAG:?Set TAG env var (e.g. beauty)}"

case "$TAG" in
  steam|yelp) ENCODE_BS=8 ;;
  *)          ENCODE_BS=128 ;;
esac

SEQREC_CKPT=outputs/seqrec_qwen3_${TAG}/best
SLICES_PARQUET=outputs/seqrec_qwen3_${TAG}/slices_filtered/per_example.parquet
PROFILE_CACHE=outputs/profiles_${TAG}_v6_cot_cf.jsonl
EMB_CACHE=outputs/emb_cache_${TAG}
ABL_DIR=outputs/fusion_${TAG}_v6cf_gated_film_filtered

echo "=== ablation film: DATASET=${DATASET} TAG=${TAG} ==="

for prereq in "${SEQREC_CKPT}/best_val_metrics.json" "${SLICES_PARQUET}" "${PROFILE_CACHE}"; do
  if [[ ! -f "${prereq}" ]]; then
    echo "missing prerequisite: ${prereq} — run run_oracle_pipeline.sh for ${TAG} first" >&2
    exit 1
  fi
done

# ============ train FiLM fusion head, no oracle aux loss ============
if [[ -f "${ABL_DIR}/fusion_head.pt" ]]; then
  echo "[skip] head already at ${ABL_DIR}/fusion_head.pt"
else
  echo "[run ] train_fusion (gated_profile_film, gate-aux-lambda 0) -> ${ABL_DIR}"
  python train_fusion.py \
    --dataset-name "${DATASET}" \
    --fusion-head-type gated_profile_film \
    --enrich-text-input \
    --filter-seen-items \
    --gate-aux-lambda 0 \
    --fusion-output-dir "${ABL_DIR}" \
    --seqrec-checkpoint "${SEQREC_CKPT}" \
    --encode-batch-size "${ENCODE_BS}" \
    --emb-cache-dir "${EMB_CACHE}" \
    --llm-profile-cache "${PROFILE_CACHE}"
fi

# ============ slice eval prof vs zero-profile (test + val) ============
for split in test val; do
  if [[ "${split}" == "test" ]]; then
    OUT_DIR=${ABL_DIR}/slices
    BL_PARQUET=${SLICES_PARQUET}
  else
    OUT_DIR=${ABL_DIR}/slices_val
    BL_PARQUET=outputs/seqrec_qwen3_${TAG}/slices_filtered_val/per_example.parquet
  fi
  PER_EX="${OUT_DIR}/per_example_prof_vs_zero.parquet"
  if [[ -f "${PER_EX}" ]]; then
    echo "[skip] ${PER_EX}"
    continue
  fi
  echo "[run ] slice_fusion (${split}) -> ${OUT_DIR}"
  python slice_fusion.py \
    --dataset-name "${DATASET}" \
    --fusion-head-path "${ABL_DIR}/fusion_head.pt" \
    --llm-profile-cache "${PROFILE_CACHE}" \
    --baseline-per-example "${BL_PARQUET}" \
    --seqrec-checkpoint "${SEQREC_CKPT}" \
    --enrich-text-input \
    --filter-seen-items \
    --split "${split}" \
    --encode-batch-size "${ENCODE_BS}" \
    --emb-cache-dir "${EMB_CACHE}" \
    --output-dir "${OUT_DIR}"
done

# ============ post-hoc analysis ============
echo "[run ] analyze_oracle_slices -> ${ABL_DIR}/slices/oracle_vs_baseline.json"
python analyze_oracle_slices.py \
  --per-example "${ABL_DIR}/slices/per_example_prof_vs_zero.parquet" \
  --output-json "${ABL_DIR}/slices/oracle_vs_baseline.json"

echo "=== ablation film DONE for ${DATASET} (${TAG}) ==="
