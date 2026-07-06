#!/bin/bash
#SBATCH --job-name=oracle_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=80G
#SBATCH --time=6-23:00:00
# Cluster policy (HPC IT, 2026-07-06): max 80G RAM with an rtx_6000 GPU, and
# do NOT set cpus-per-task — the per-GPU default applies.

# Full end-to-end oracle-R@10 pipeline for one domain.
# Required env vars:
#   DATASET  dataset name (e.g. Toys_and_Games, MovieLens-1M, Steam, Yelp), must exist in data.DATASET_REGISTRY
#   TAG      short slug used in output paths (e.g. toys, sports, home, ml1m, steam, yelp)

source ~/.bashrc
conda activate my_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Stream python output into the SLURM log; otherwise progress is buffered and
# lost when a step is OOM-killed.
export PYTHONUNBUFFERED=1
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
# vLLM forks EngineCore subprocesses; main process may already have CUDA
# initialized (TF import in our deps), which makes 'fork' fail. 'spawn'
# starts a clean Python process for the worker.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

set -euo pipefail

: "${DATASET:?Set DATASET env var (e.g. Toys_and_Games)}"
: "${TAG:?Set TAG env var (e.g. toys)}"

# Steam/Yelp have dense, multilingual reviews; smaller per-item review budget keeps
# profile prompts under the LLM context window. Amazon keeps the default 40.
case "$TAG" in
  steam|yelp) REVIEW_WORDS=8 ;;
  *)          REVIEW_WORDS=40 ;;
esac

# Steam/Yelp histories are long and the seqrec encoder does not truncate
# (max_seq_length=32768), so the default encode batch of 128 OOMs a 24GB GPU
# during train_fusion's one-shot pre-encoding pass. Use a smaller batch.
case "$TAG" in
  steam|yelp) ENCODE_BS=8 ;;
  *)          ENCODE_BS=128 ;;
esac

SEQREC_DIR=outputs/seqrec_qwen3_${TAG}
SEQREC_CKPT=${SEQREC_DIR}/best
SLICES_DIR=${SEQREC_DIR}/slices_filtered
SLICES_PARQUET=${SLICES_DIR}/per_example.parquet
COOCCUR=outputs/cooccur_${TAG}.npz
CF_CTX=outputs/cf_context_${TAG}.jsonl
USER_CF=outputs/user_cf_emb_${TAG}.npz
PROFILE_CACHE=outputs/profiles_${TAG}_v6_cot_cf.jsonl
ORACLE_DIR=outputs/fusion_${TAG}_v6cf_gated_oracle_filtered_r10
# Sharded on-disk embedding cache: a killed/OOM'd train_fusion resumes encoding
# instead of redoing it, and ablation runs reuse the same embeddings.
EMB_CACHE=outputs/emb_cache_${TAG}

echo "=== oracle pipeline: DATASET=${DATASET} TAG=${TAG} ==="

# ============ Step 1: Qwen3 seqrec backbone ============
if [[ -f "${SEQREC_CKPT}/best_val_metrics.json" ]]; then
  echo "[skip] backbone already at ${SEQREC_CKPT}"
else
  echo "[run ] training Qwen3 backbone -> ${SEQREC_DIR}"
  python train.py \
    --dataset-name "${DATASET}" \
    --model-name Qwen/Qwen3-Embedding-0.6B \
    --enrich-text-input \
    --bf16 \
    --output-dir "${SEQREC_DIR}"
fi

# ============ Step 2: CF artifacts (cooccur, context, user SVD) ============
if [[ -f "${COOCCUR}" ]]; then
  echo "[skip] ${COOCCUR}"
else
  echo "[run ] build_cooccur -> ${COOCCUR}"
  python build_cooccur.py --dataset-name "${DATASET}" --output "${COOCCUR}"
fi

if [[ -f "${CF_CTX}" ]]; then
  echo "[skip] ${CF_CTX}"
else
  echo "[run ] build_cf_context -> ${CF_CTX}"
  python build_cf_context.py --dataset-name "${DATASET}" --output-path "${CF_CTX}"
fi

if [[ -f "${USER_CF}" ]]; then
  echo "[skip] ${USER_CF}"
else
  echo "[run ] build_user_cf_emb -> ${USER_CF}"
  python build_user_cf_emb.py --dataset-name "${DATASET}" --output-path "${USER_CF}"
fi

# ============ Step 3: v6 CoT+CF profile cache ============
if [[ -f "${PROFILE_CACHE}" ]]; then
  echo "[skip] profile cache already at ${PROFILE_CACHE} (rerun will resume if incomplete)"
fi
echo "[run ] generate_profiles -> ${PROFILE_CACHE}"
python generate_profiles.py \
  --dataset-name "${DATASET}" \
  --llm-profile-cache "${PROFILE_CACHE}" \
  --cf-context-cache "${CF_CTX}" \
  --llm-profile-review-words "${REVIEW_WORDS}" \
  --use-vllm

# ============ Step 4: filtered baseline slices ============
if [[ -f "${SLICES_PARQUET}" ]]; then
  echo "[skip] ${SLICES_PARQUET}"
else
  echo "[run ] evaluate_slices -> ${SLICES_DIR}"
  python evaluate_slices.py \
    --dataset-name "${DATASET}" \
    --seqrec-checkpoint "${SEQREC_CKPT}" \
    --enrich-text-input \
    --eval-catalog interacted \
    --filter-seen-items \
    --output-dir "${SLICES_DIR}"
fi

# ============ Step 5: train R@10-targeted oracle ============
if [[ -f "${ORACLE_DIR}/fusion_head.pt" ]]; then
  echo "[skip] oracle head already at ${ORACLE_DIR}/fusion_head.pt"
else
  echo "[run ] train_fusion (oracle r10) -> ${ORACLE_DIR}"
  python train_fusion.py \
    --dataset-name "${DATASET}" \
    --fusion-head-type gated_profile \
    --enrich-text-input \
    --filter-seen-items \
    --gate-aux-objective r10_recovery \
    --gate-aux-pos-weight balanced \
    --gate-aux-lambda 0.3 \
    --gate-aux-alpha 0.3 \
    --fusion-output-dir "${ORACLE_DIR}" \
    --seqrec-checkpoint "${SEQREC_CKPT}" \
    --encode-batch-size "${ENCODE_BS}" \
    --emb-cache-dir "${EMB_CACHE}" \
    --llm-profile-cache "${PROFILE_CACHE}"
fi

# ============ Step 6: slice eval prof vs zero-profile ============
PER_EX="${ORACLE_DIR}/slices/per_example_prof_vs_zero.parquet"
if [[ -f "${PER_EX}" ]]; then
  echo "[skip] ${PER_EX}"
else
  echo "[run ] slice_fusion -> ${ORACLE_DIR}/slices"
  python slice_fusion.py \
    --dataset-name "${DATASET}" \
    --fusion-head-path "${ORACLE_DIR}/fusion_head.pt" \
    --llm-profile-cache "${PROFILE_CACHE}" \
    --baseline-per-example "${SLICES_PARQUET}" \
    --seqrec-checkpoint "${SEQREC_CKPT}" \
    --enrich-text-input \
    --filter-seen-items \
    --output-dir "${ORACLE_DIR}/slices"
fi

# ============ Step 7: post-hoc analysis ============
echo "[run ] analyze_oracle_slices -> ${ORACLE_DIR}/slices/oracle_vs_baseline.json"
python analyze_oracle_slices.py \
  --per-example "${PER_EX}" \
  --output-json "${ORACLE_DIR}/slices/oracle_vs_baseline.json"

echo "=== oracle pipeline DONE for ${DATASET} (${TAG}) ==="
