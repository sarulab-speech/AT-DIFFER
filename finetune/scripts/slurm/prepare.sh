#!/bin/bash
#SBATCH --job-name=at-differ-prepare
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
# Create the 16 kHz audio cache and the train / eval manifests. No GPU is used.
#
# Submit from finetune/ after `uv sync`:
#   sbatch scripts/slurm/prepare.sh <dataset-dir>
#
# <dataset-dir> is the output directory of the dataset stage (containing
# train/ and val/). The cache (about 30 GB) and the manifests are written under
# ../outputs/finetune/, where configs/method*.yaml expect them. Files already
# in the cache are skipped, so the same command can be submitted again if the
# job is interrupted.

set -euo pipefail

DATASET_DIR="${1:?usage: sbatch scripts/slurm/prepare.sh <dataset-dir>}"

cd "${SLURM_SUBMIT_DIR:-.}"

uv run at-differ-finetune prepare \
  --dataset-dir "${DATASET_DIR}" \
  --cache-dir ../outputs/finetune/cache \
  --manifest-dir ../outputs/finetune/manifests \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
