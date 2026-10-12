#!/bin/bash
#SBATCH --job-name=at-differ-merge
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Merge a trained adapter into the base model, for the inference stage. The
# merge runs on the CPU (no GPU) and builds the whole 30B model in memory
# (about 70 GB at peak). The output is about 61 GB.
#
# Submit from finetune/:
#   sbatch scripts/slurm/merge.sh <adapter-dir> <output-dir>
#   e.g. sbatch scripts/slurm/merge.sh ../outputs/finetune/method2/adapter ../outputs/finetune/method2/merged

set -euo pipefail

ADAPTER="${1:?usage: sbatch scripts/slurm/merge.sh <adapter-dir> <output-dir>}"
OUT="${2:?usage: sbatch scripts/slurm/merge.sh <adapter-dir> <output-dir>}"

cd "${SLURM_SUBMIT_DIR:-.}"

if [ ! -f "${ADAPTER}/adapter_config.json" ]; then
  echo "No adapter_config.json in ${ADAPTER}. Has training finished?" >&2
  exit 1
fi

uv run at-differ-finetune merge --adapter "${ADAPTER}" --out "${OUT}"
