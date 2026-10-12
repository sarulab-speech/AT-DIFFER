#!/bin/bash
#SBATCH --job-name=at-differ-train
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#
# Train one method on two GPUs (the paper used one node with two
# RTX PRO 6000 Blackwell Max-Q 96 GB).
#
# Submit from finetune/ after `uv sync` and `prepare`:
#   sbatch scripts/slurm/train.sh configs/method2.yaml
#
# Each of the paper's runs took about 78 hours. If the job reaches its time
# limit, submit the same command again: training resumes from the latest
# checkpoint in output_dir (saved every 250 steps).

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/slurm/train.sh <config>}"

cd "${SLURM_SUBMIT_DIR:-.}"

# torchrun's default rendezvous port (29500) is fixed, so two training jobs on
# the same node would collide. Derive the port from the job ID instead.
MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"

# Avoid the warning about tokenizer parallelism in forked DataLoader workers.
export TOKENIZERS_PARALLELISM=false

nvidia-smi -L || true
echo "config=${CONFIG} gpus=${NUM_GPUS} master_port=${MASTER_PORT}"

uv run torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  -m at_differ_finetune.train --config "${CONFIG}"
