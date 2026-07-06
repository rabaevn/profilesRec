#!/bin/bash
# Submit run_oracle_pipeline.sh for one or more domains.
#
# Usage:
#   ./submit_oracle_domain.sh                 # submits toys only (smoke)
#   ./submit_oracle_domain.sh toys            # same
#   ./submit_oracle_domain.sh sports home     # fan-out (after toys validated)
#   ./submit_oracle_domain.sh all             # all Amazon domains (toys sports home)
#   ./submit_oracle_domain.sh ml1m steam yelp # non-Amazon datasets
#   ./submit_oracle_domain.sh all-extra       # the three non-Amazon datasets

set -euo pipefail

declare -A DOMAIN_OF=(
  [beauty]=Beauty
  [toys]=Toys_and_Games
  [sports]=Sports_and_Outdoors
  [home]=Home_and_Kitchen
  [ml1m]=MovieLens-1M
  [steam]=Steam
  [yelp]=Yelp
)

if [[ $# -eq 0 ]]; then
  TAGS=(toys)
elif [[ "$1" == "all" ]]; then
  TAGS=(toys sports home)
elif [[ "$1" == "all-extra" ]]; then
  TAGS=(ml1m steam yelp)
else
  TAGS=("$@")
fi

for tag in "${TAGS[@]}"; do
  dataset="${DOMAIN_OF[$tag]:-}"
  if [[ -z "$dataset" ]]; then
    echo "unknown tag: $tag  (expected one of: ${!DOMAIN_OF[*]} | all)" >&2
    exit 1
  fi
  # Cluster policy: 80G max with rtx_6000 (set in run_oracle_pipeline.sh).
  # Steam/Yelp fit via train_fusion's emb cache + in-run memory frees.
  echo "submitting oracle_${tag}  DATASET=${dataset}"
  sbatch \
    --job-name="oracle_${tag}" \
    --export="ALL,DATASET=${dataset},TAG=${tag}" \
    run_oracle_pipeline.sh
done
