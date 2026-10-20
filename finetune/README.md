# AT-DIFFER fine-tuning

This directory fine-tunes
[Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
on AT-DIFFER with LoRA, as in Methods 1–3 of the paper, and merges the trained
adapter into the base model so that the
[inference stage](../inference/README.md) can load it.

For each audio–text pair, the model receives the audio and the prompt
[`../resources/prompt.txt`](../resources/prompt.txt) with the text inserted,
and is trained to output three lines:

```
DISCREPANCY: 1
TYPES: event_omission, phantom_sound
EXPLANATION: The caption mistakes the sound of rain for the sound of applause.
```

The type names on the TYPES line are those of the prompt, which differ from
the names of the difference types in the paper and the metadata. See
[Dataset](../README.md#dataset) in the top-level README for how they
correspond.

LoRA is applied to the query, key, value, and output projections of the
self-attention layers of the language model. The audio encoder and the
Mixture-of-Experts feed-forward layers are frozen.

## Methods

The three methods differ only in which lines are supervised and how their
losses are combined. In all of them, the token-level loss is averaged within
each audio–text pair and then across pairs.

| Method | Config | Supervised lines | Loss |
|---|---|---|---|
| Method 1 (Difference caption only) | `configs/method1.yaml` | EXPLANATION text | Cross-entropy over the difference caption |
| Method 2 (Concatenate three tasks) | `configs/method2.yaml` | DISCREPANCY, TYPES, EXPLANATION | One cross-entropy over all tokens of the three lines (`loss_reduction: concatenate`) |
| Method 3 (Separate three tasks) | `configs/method3.yaml` | DISCREPANCY, TYPES, EXPLANATION | Cross-entropy of each line, combined with equal weights of 1/3 (`loss_reduction: separate`) |

The configs hold the settings of the paper's runs, including those not
stated in the paper:

- 1 epoch over all 149,172 train items (4,662 steps).
- Batch size 1 per GPU × gradient accumulation 16 × 2 GPUs = 32.
- Learning rate 1.0 × 10⁻⁴, cosine schedule, warmup ratio 0.03, weight
  decay 0, maximum gradient norm 1.0, seed 0.
- LoRA rank 16, α = 32, dropout 0.05.
- bf16, gradient checkpointing, SDPA attention.
- Batches grouped by audio duration. Audio longer than 40 s is truncated.
- The eval loss on 500 val items (`configs/eval_ids.txt`) is computed every
  250 steps, when a checkpoint is also saved. The weights after the last step
  are used.

An empty thinking block (`<think>\n\n</think>\n\n`) is placed at the start of
the model response. Use the model key `qwen3-omni-nothink` of the inference
stage for the fine-tuned models, which does the same.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the pinned
  dependencies into `finetune/.venv`.
- Linux with two NVIDIA GPUs and a driver that supports CUDA 13.0. The paper
  used one node with two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation
  Edition GPUs (96 GB each). The merge runs on the CPU and needs about 70 GB
  of memory.
- The train and val splits of AT-DIFFER, created by the
  [dataset stage](../dataset/README.md).
- Disk space: 63 GB for the base model, about 32 GB for the 16 kHz audio
  cache, and 63 GB for each merged checkpoint.

## Usage

All commands are run from `finetune/`. Paths in the configs are relative to
this directory, and outputs go to `../outputs/`.

### 1. Set up

```bash
cd finetune
uv sync
```

Download the base model at the revision used in the paper:

```bash
uv run hf download Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --revision 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b \
    --local-dir ../outputs/models/Qwen3-Omni-30B-A3B-Thinking
```

The configs (`model_path`) and the inference stage read the base model from
this directory. To keep it elsewhere, change `model_path` in the configs.

### 2. Prepare the audio cache and the manifests

```bash
uv run at-differ-finetune prepare \
    --dataset-dir  /path/to/AT-DIFFER \
    --cache-dir    ../outputs/finetune/cache \
    --manifest-dir ../outputs/finetune/manifests
```

| Argument | Description |
|---|---|
| `--dataset-dir` | Output directory of the dataset stage (containing `train/` and `val/`). |
| `--cache-dir` | Directory for the 16 kHz audio cache. |
| `--manifest-dir` | Directory to write `train.jsonl` and `eval.jsonl` to. The configs read them from `../outputs/finetune/manifests`. |
| `--workers` | Number of parallel conversion processes (default: 8). |

The audio of the dataset (22.05 kHz, 32-bit float) is converted once to
16 kHz, 16-bit PCM WAV files, which are read during training. Only the audio
used for training is converted: all of train and the 500 eval items of val.
Files already in the cache are skipped, so running the command again after an
interruption continues where it stopped.

The manifests list the training and eval items in the order used for the
paper: for train, all items (those with a difference followed by those
without, in metadata order) shuffled with seed 0; for eval, the 500 val items
of `configs/eval_ids.txt`.

### 3. Train

```bash
uv run torchrun --nproc_per_node 2 -m at_differ_finetune.train --config configs/method2.yaml
```

Replace `method2` with `method1` or `method3` for the other methods.
`uv run at-differ-finetune train --config ...` runs the same training in a
single process.

If the output directory already has checkpoints, training resumes from the
latest one, so an interrupted run can be continued with the same command.

### 4. Merge

```bash
uv run at-differ-finetune merge \
    --adapter ../outputs/finetune/method2/adapter \
    --out     ../outputs/finetune/method2/merged
```

This merges the adapter into the base model and saves the result as an
ordinary checkpoint, which the inference stage loads with
`--model-path ../outputs/finetune/method2/merged`. By default, the base model
is the one recorded in the adapter (the `model_path` used for training);
`--base-model` overrides it.

After saving, the command checks that the weights were written completely and
that the checkpoint has the same model type, architecture, and weight names
as the base model.

### Running on Slurm

`scripts/slurm/` has examples of job scripts. Submit them from `finetune/`
after `uv sync`:

```bash
sbatch scripts/slurm/prepare.sh /path/to/AT-DIFFER
sbatch scripts/slurm/train.sh configs/method2.yaml
sbatch scripts/slurm/merge.sh ../outputs/finetune/method2/adapter ../outputs/finetune/method2/merged
```

Adjust the `#SBATCH` lines (partition, memory, time limit) to your cluster.
If a training job reaches its time limit, submit it again to resume.

## Output

```
outputs/
├── models/Qwen3-Omni-30B-A3B-Thinking/   # base model (step 1)
└── finetune/
    ├── cache/{train,val}/audio_*.wav     # 16 kHz audio cache
    ├── manifests/{train,eval}.jsonl
    └── method2/
        ├── checkpoint-*/                 # every 250 steps
        ├── adapter/                      # final LoRA adapter
        ├── train_config.json             # settings of the run
        ├── log_history.json              # training and eval losses
        └── merged/                       # merged checkpoint (step 4)
```

## Reproducibility

The dependency versions in `uv.lock` are those of the environment used for
the paper's training. The manifests created by `prepare` were verified to be
identical to those used for the paper, in the order of the items and in every
field. Training on GPUs is not bit-for-bit deterministic, so the trained
weights may differ slightly from ours.

## Tests

```bash
cd finetune
uv run pytest
```

The tests need neither a GPU nor the model. On macOS, `uv sync` installs the
CPU build of PyTorch, so the tests also run there.
