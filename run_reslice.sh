#!/bin/bash
#SBATCH --job-name=reslice
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
# Cluster policy (HPC IT, 2026-07-06): max 80G RAM with an rtx_6000 GPU, and
# do NOT set cpus-per-task — the per-GPU default applies.

# Re-slice one domain under the unified eval protocol (ranking.py): recompute
# the text-only baseline slices and every existing fusion head's prof-vs-zero
# slices, on test AND val, sharing the emb cache. No retraining — the heads
# are finished artifacts; only the eval protocol changed (evaluate_slices'
# gold-score off-by-one fix + shared rank path + cached embeddings).
# Old outputs are backed up by rename, never deleted.
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

BAK_SUFFIX=bak$(date +%Y%m%d)
SEQREC_DIR=outputs/seqrec_qwen3_${TAG}
SEQREC_CKPT=${SEQREC_DIR}/best
PROFILE_CACHE=outputs/profiles_${TAG}_v6_cot_cf.jsonl
EMB_CACHE=outputs/emb_cache_${TAG}
# Fusion-head variants to re-slice; missing dirs are skipped.
HEAD_VARIANTS=(
  "fusion_${TAG}_v6cf_gated_oracle_filtered_r10"
  "fusion_${TAG}_v6cf_gated_nooracle_filtered"
  "fusion_${TAG}_v6cf_gated_film_filtered"
)

backup() {  # backup <path>: rename out of the way so the existence gate reruns
  local path=$1
  if [[ -e "${path}" ]]; then
    local dst="${path}_${BAK_SUFFIX}"
    local n=1
    while [[ -e "${dst}" ]]; do dst="${path}_${BAK_SUFFIX}.${n}"; n=$((n+1)); done
    echo "[bak ] ${path} -> ${dst}"
    mv "${path}" "${dst}"
  fi
}

echo "=== re-slice (unified protocol): DATASET=${DATASET} TAG=${TAG} ==="

for split in test val; do
  if [[ "${split}" == "test" ]]; then
    SLICES_DIR=${SEQREC_DIR}/slices_filtered
  else
    SLICES_DIR=${SEQREC_DIR}/slices_filtered_val
  fi
  backup "${SLICES_DIR}"
  echo "[run ] evaluate_slices (${split}) -> ${SLICES_DIR}"
  python evaluate_slices.py \
    --dataset-name "${DATASET}" \
    --seqrec-checkpoint "${SEQREC_CKPT}" \
    --enrich-text-input \
    --eval-catalog interacted \
    --filter-seen-items \
    --split "${split}" \
    --encode-batch-size "${ENCODE_BS}" \
    --emb-cache-dir "${EMB_CACHE}" \
    --output-dir "${SLICES_DIR}"

  for variant in "${HEAD_VARIANTS[@]}"; do
    HEAD_DIR=outputs/${variant}
    if [[ ! -f "${HEAD_DIR}/fusion_head.pt" ]]; then
      echo "[skip] ${HEAD_DIR} (no fusion_head.pt)"
      continue
    fi
    if [[ "${split}" == "test" ]]; then
      OUT_DIR=${HEAD_DIR}/slices
    else
      OUT_DIR=${HEAD_DIR}/slices_val
    fi
    backup "${OUT_DIR}"
    echo "[run ] slice_fusion (${split}) -> ${OUT_DIR}"
    python slice_fusion.py \
      --dataset-name "${DATASET}" \
      --fusion-head-path "${HEAD_DIR}/fusion_head.pt" \
      --llm-profile-cache "${PROFILE_CACHE}" \
      --baseline-per-example "${SLICES_DIR}/per_example.parquet" \
      --seqrec-checkpoint "${SEQREC_CKPT}" \
      --enrich-text-input \
      --filter-seen-items \
      --split "${split}" \
      --encode-batch-size "${ENCODE_BS}" \
      --emb-cache-dir "${EMB_CACHE}" \
      --output-dir "${OUT_DIR}"
    if [[ "${split}" == "test" ]]; then
      echo "[run ] analyze_oracle_slices -> ${OUT_DIR}/oracle_vs_baseline.json"
      python analyze_oracle_slices.py \
        --per-example "${OUT_DIR}/per_example_prof_vs_zero.parquet" \
        --output-json "${OUT_DIR}/oracle_vs_baseline.json"
    fi
  done
done

echo "=== re-slice DONE for ${DATASET} (${TAG}) ==="
