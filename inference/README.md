# AT-DIFFER inference

This directory runs
[Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
(the Baseline of the paper) and the fine-tuned models (Methods 1–3) on the
test split of AT-DIFFER with [vLLM](https://github.com/vllm-project/vllm), and
saves their outputs for evaluation.

For each audio–text pair, the model receives the audio and the prompt
[`../resources/prompt.txt`](../resources/prompt.txt) with the text inserted,
and is asked to output three lines:

```
DISCREPANCY: 1
TYPES: event_omission, phantom_sound
EXPLANATION: The caption mistakes the sound of rain for the sound of applause.
```

The type names on the TYPES line are those of the prompt, which differ from
the names of the difference types in the paper and the metadata. See
[Dataset](../README.md#dataset) in the top-level README for how they
correspond.

## Models

The model is chosen with `--model` and `--model-path`:

| Paper | `--model` | `--model-path` |
|---|---|---|
| Baseline | `qwen3-omni-thinking` (default) | The base model (default: `../outputs/models/Qwen3-Omni-30B-A3B-Thinking`) |
| Method 1 (Difference caption only) | `qwen3-omni-nothink` | Merged checkpoint of Method 1 from the [finetune stage](../finetune/README.md) |
| Method 2 (Concatenate three tasks) | `qwen3-omni-nothink` | Merged checkpoint of Method 2 |
| Method 3 (Separate three tasks) | `qwen3-omni-nothink` | Merged checkpoint of Method 3 |

- `qwen3-omni-thinking` lets the model think before answering: its output is
  a `<think>…</think>` block followed by the three lines, unless it reaches
  the limit of 4,096 output tokens before the block is closed. **The Baseline
  results in the paper were obtained with thinking enabled in this way.**
- `qwen3-omni-nothink` places an empty thinking block
  (`<think>\n\n</think>\n\n`) at the start of the model response, as in
  fine-tuning, so that the model writes the three lines directly. Use it only
  with the fine-tuned models.

The other settings are the same for all models, and the defaults of the
command are the settings of the paper's runs:

- vLLM: tensor parallel size 2, GPU memory utilization 0.9, maximum model
  length 32,768, at most 16 sequences at a time, eager mode, bf16.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, at most 4,096 output
  tokens.
- The items are passed to vLLM in chunks of 64.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the pinned
  dependencies into `inference/.venv`. The environment can be created only on
  Linux x86_64, because vLLM and decord have no packages for other platforms.
- Two NVIDIA GPUs and a driver that supports CUDA 13.0. The paper used one
  node with two NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs
  (96 GB each).
- **On Blackwell GPUs, the CUDA compiler `nvcc` on the GPU node.** vLLM uses a
  Mixture-of-Experts kernel of FlashInfer that is compiled at run time for
  these GPUs. If `nvcc` is not found (through `CUDA_HOME`, on `PATH`, or
  under `/usr/local/cuda`), the run fails at startup with
  `Could not find nvcc`.
- The test split of AT-DIFFER, created by the
  [dataset stage](../dataset/README.md).
- The base model (63 GB) for the Baseline, and the merged checkpoints from
  the [finetune stage](../finetune/README.md) for Methods 1–3.
- Network access on the first run. The dependency `static-ffmpeg` downloads
  ffmpeg binaries from GitHub into the environment the first time it is used.
  If the GPU nodes have no network access, run `uv run static_ffmpeg_paths`
  once on a machine that has (for example, the login node), after `uv sync`.

## Usage

All commands are run from `inference/`, and outputs go to `../outputs/`.

### 1. Set up

```bash
cd inference
uv sync
```

The Baseline reads the base model from `../outputs/models/Qwen3-Omni-30B-A3B-Thinking`.
If you have already downloaded it for the finetune stage, nothing more is
needed. Otherwise, download it at the revision used in the paper:

```bash
uv run hf download Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --revision 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b \
    --local-dir ../outputs/models/Qwen3-Omni-30B-A3B-Thinking
```

### 2. Run

Set `FORCE_QWENVL_VIDEO_READER=decord`, as in the paper's runs:

```bash
export FORCE_QWENVL_VIDEO_READER=decord
```

Baseline:

```bash
uv run at-differ-inference \
    --data-dir   /path/to/AT-DIFFER/test \
    --output-dir ../outputs/inference/baseline
```

Methods 1–3 (here, Method 2):

```bash
uv run at-differ-inference \
    --data-dir   /path/to/AT-DIFFER/test \
    --output-dir ../outputs/inference/method2 \
    --model      qwen3-omni-nothink \
    --model-path ../outputs/finetune/method2/merged
```

| Argument | Description |
|---|---|
| `--data-dir` | Directory of the test split (containing `metadata.csv` and `audios/`). |
| `--output-dir` | Directory under which a run directory is created. |
| `--model` | `qwen3-omni-thinking` (default) or `qwen3-omni-nothink` (see [Models](#models)). |
| `--model-path` | The base model (default: `../outputs/models/Qwen3-Omni-30B-A3B-Thinking`) or a merged checkpoint. |
| `--prompt-file` | Prompt template (default: `../resources/prompt.txt`). |
| `--offset`, `--limit` | Skip the first `N` items, and process at most `N` items. By default, all items are processed. |
| `--note` | Free-form note recorded in `run_metadata.yaml`. |

The vLLM and sampling settings can also be changed with options (see
`uv run at-differ-inference --help`); their defaults are those of the paper.

The results are appended to `predictions.csv` after each chunk of 64 items,
so the results of the finished chunks are kept if a run is interrupted. A new run
always starts from the first item (after `--offset`) in a new run directory.

### Running on Slurm

`scripts/slurm/infer.sh` is an example of a job script. Submit it from
`inference/` after `uv sync`, with the arguments of `at-differ-inference`:

```bash
sbatch scripts/slurm/infer.sh --data-dir /path/to/AT-DIFFER/test \
    --output-dir ../outputs/inference/baseline
sbatch scripts/slurm/infer.sh --data-dir /path/to/AT-DIFFER/test \
    --output-dir ../outputs/inference/method2 \
    --model qwen3-omni-nothink --model-path ../outputs/finetune/method2/merged
```

The script sets `FORCE_QWENVL_VIDEO_READER=decord` and sets the tensor
parallel size to the number of GPUs allocated by Slurm. Adjust the `#SBATCH`
lines (partition, memory, time limit) to your cluster.

## Output

```
outputs/inference/baseline/
└── <YYYYMMDDhhmm>/            # run directory, named after the start time
    ├── predictions.csv
    └── run_metadata.yaml
```

`predictions.csv` has one row per item, in the order of `metadata.csv`:

| Column | Description |
|---|---|
| `id`, `audio_structure`, `structure_params`, `difference_types`, `text`, `difference_caption` | Copied from `metadata.csv` (see [Metadata columns](../dataset/README.md#metadata-columns) in the dataset README). |
| `audio_path` | Absolute path of the audio file. |
| `model_output` | Output of the model, with leading and trailing whitespace removed. With `qwen3-omni-thinking`, it includes the `<think>…</think>` block. If a chunk failed, its rows hold `[ERROR] <exception>` instead. |

`run_metadata.yaml` records the settings (vLLM and sampling settings, the
model, and the full prompt), the paths of the data and the output, the number
of processed and failed items, the start and end times, the command line, and
the Python version.

## Reproducibility

The dependency versions in `uv.lock` are those of the environment used for
the paper's inference runs. The code was verified to give the model the same
inputs (prompt, audio, and settings) as the code used for those runs.
However, the outputs are sampled (no seed is set, so vLLM's default is
used), and the batching of vLLM and computation on GPUs are not
deterministic, so the outputs may differ from ours even with the same
settings.
