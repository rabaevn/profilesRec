#!/bin/bash
#SBATCH --job-name=train_oracle_r10
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate my_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"

set -euo pipefail

ORACLE_DIR=outputs/fusion_beauty_v6cf_gated_oracle_filtered_r10

# ============ Step 1: train R@10-targeted oracle ============
# Label rule: y=1 iff text-only misses top-10 AND profile-mix recovers (rank<=10).
# Sparse positive class -> pos_weight=balanced (n_neg/n_pos).
# 3x previous lambda to compensate for sparser supervision.
python train_fusion.py \
  --fusion-head-type gated_profile \
  --enrich-text-input \
  --filter-seen-items \
  --gate-aux-objective r10_recovery \
  --gate-aux-pos-weight balanced \
  --gate-aux-lambda 0.3 \
  --gate-aux-alpha 0.3 \
  --fusion-output-dir "$ORACLE_DIR" \
  --seqrec-checkpoint outputs/seqrec_qwen3_beauty/best \
  --llm-profile-cache outputs/profiles_beauty_v6_cot_cf.jsonl

# ============ Step 2: slice eval against filtered baseline ============
python slice_fusion.py \
  --fusion-head-path  "$ORACLE_DIR/fusion_head.pt" \
  --llm-profile-cache outputs/profiles_beauty_v6_cot_cf.jsonl \
  --baseline-per-example outputs/seqrec_qwen3_beauty/slices_filtered/per_example.parquet \
  --seqrec-checkpoint outputs/seqrec_qwen3_beauty/best \
  --enrich-text-input \
  --filter-seen-items \
  --output-dir "$ORACLE_DIR/slices"

# ============ Step 3: post-hoc analysis ============
python analyze_oracle_slices.py \
  --per-example "$ORACLE_DIR/slices/per_example_prof_vs_zero.parquet" \
  --output-json "$ORACLE_DIR/slices/oracle_vs_baseline.json"
