#!/bin/bash
#SBATCH --job-name=at-differ-infer
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#
# Run inference on two GPUs (the paper used one node with two
# RTX PRO 6000 Blackwell Max-Q 96 GB). The arguments are passed to
# at-differ-inference, and the tensor parallel size follows the number of
# GPUs allocated by Slurm.
#
# Submit from inference/ after `uv sync`:
#   Baseline:
#     sbatch scripts/slurm/infer.sh --data-dir /path/to/AT-DIFFER/test \
#         --output-dir ../outputs/inference/baseline
#   Methods 1-3 (a merged checkpoint from the finetune stage):
#     sbatch scripts/slurm/infer.sh --data-dir /path/to/AT-DIFFER/test \
#         --output-dir ../outputs/inference/method2 \
#         --model qwen3-omni-nothink --model-path ../outputs/finetune/method2/merged
#
# On the test split, the paper's runs took about 24 hours for the Baseline
# (85,510 s; it writes a long thinking block for every item) and 1.4-1.9 hours
# for each of Methods 1-3 (4,935-6,723 s).

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: sbatch scripts/slurm/infer.sh --data-dir <dir> --output-dir <dir> [options of at-differ-inference]" >&2
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-.}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"

# Set as in the paper's runs.
export FORCE_QWENVL_VIDEO_READER=decord

# On Blackwell GPUs, FlashInfer compiles its MoE kernel with nvcc at run time,
# so the job fails on a node without nvcc (see README).
nvidia-smi -L || true
echo "gpus=${NUM_GPUS} nvcc=$(command -v nvcc || echo 'not found') CUDA_HOME=${CUDA_HOME:-unset}"

uv run at-differ-inference --tensor-parallel-size "${NUM_GPUS}" "$@"
