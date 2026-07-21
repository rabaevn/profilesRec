#!/bin/bash
# Submit an ablation run script for one or more domains.
#
# Usage:
#   ./submit_ablation.sh no_oracle beauty toys sports steam
#
# First arg <name> maps to run_ablation_<name>.sh; remaining args are tags.

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

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <ablation-name> <tag> [tag ...]" >&2
  exit 1
fi

ABLATION="$1"; shift
RUN_SCRIPT="run_ablation_${ABLATION}.sh"
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "no such ablation script: ${RUN_SCRIPT}" >&2
  exit 1
fi

for tag in "$@"; do
  dataset="${DOMAIN_OF[$tag]:-}"
  if [[ -z "$dataset" ]]; then
    echo "unknown tag: $tag  (expected one of: ${!DOMAIN_OF[*]})" >&2
    exit 1
  fi
  echo "submitting abl_${ABLATION}_${tag}  DATASET=${dataset}"
  sbatch \
    --job-name="abl_${ABLATION}_${tag}" \
    --export="ALL,DATASET=${dataset},TAG=${tag}" \
    "${RUN_SCRIPT}"
done
