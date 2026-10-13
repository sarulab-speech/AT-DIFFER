"""Entry point for running audio+text difference captioning inference.

Feeds each (audio, text) pair from a dataset to a multimodal LLM and saves
the model's described differences to predictions.csv, alongside run_metadata.yaml.

Usage (from inference/):
    uv run at-differ-inference \\
        --data-dir   /path/to/AT-DIFFER/test \\
        --output-dir /path/to/output \\
        [--model        qwen3-omni-thinking] \\
        [--model-path   ../outputs/models/Qwen3-Omni-30B-A3B-Thinking] \\
        [--tensor-parallel-size 2] [--gpu-memory-utilization 0.90] \\
        [--max-model-len 32768] [--max-num-seqs 16] [--max-tokens 4096] \\
        [--temperature 0.6] [--top-p 0.95] [--top-k 20] \\
        [--prompt-file ../resources/prompt.txt] \\
        [--chunk-size 64] [--offset N] [--limit N] [--note "free-form memo"]

The defaults are the settings of the paper's runs.

The dataset directory must contain:
    audios/         WAV files
    metadata.csv    rows with at least: id, audio_path, text, difference_caption
"""

import argparse
from pathlib import Path

from at_differ_inference.data.dataset import InferenceDataset
from at_differ_inference.io.run_metadata import RunMetadata
from at_differ_inference.io.saver import OutputSaver
from at_differ_inference.models.registry import build_model, registered_models
from at_differ_inference.pipeline.prompt import DEFAULT_PROMPT_FILE, PromptBuilder
from at_differ_inference.pipeline.runner import InferenceRunner

# Local directory of the base model (see README for how to download it), the
# same as in the finetune stage. Relative to the stage directory (inference/).
DEFAULT_MODEL_PATH = "../outputs/models/Qwen3-Omni-30B-A3B-Thinking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="at-differ-inference",
        description="Run audio+text difference captioning inference and save results.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Dataset directory containing audios/ and metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory under which a YYYYMMDDhhmm run folder is created.",
    )

    # --- Model selection & config -------------------------------------------
    parser.add_argument(
        "--model",
        default="qwen3-omni-thinking",
        choices=registered_models(),
        help="Registered model backend to use.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Model weights: the base model, or a merged checkpoint from the "
        "finetune stage (default: %(default)s).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="vLLM tensor parallel size (should match the number of GPUs).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM max model (context) length.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="vLLM max concurrent sequences (internal batching width).",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)

    # --- Data / prompt / run options ----------------------------------------
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Path to a prompt template file (must contain '{caption}'; "
        "default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Records per inference chunk (checkpoint granularity). "
        "Should be >= --max-num-seqs for good throughput.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N records before processing (0-based start index).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N records (after applying --offset).",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Free-form memo recorded in run_metadata.yaml.",
    )

    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error(f"--data-dir does not exist: {args.data_dir}")
    if not args.prompt_file.exists():
        parser.error(f"--prompt-file does not exist: {args.prompt_file}")
    if args.chunk_size < 1:
        parser.error(f"--chunk-size must be >= 1: {args.chunk_size}")
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be >= 1: {args.limit}")
    if args.offset < 0:
        parser.error(f"--offset must be >= 0: {args.offset}")

    return args


def build_model_config(args: argparse.Namespace) -> dict:
    return {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "model_path": args.model_path,
    }


def main() -> None:
    args = parse_args()

    metadata = RunMetadata()
    metadata.start()

    dataset = InferenceDataset(
        args.data_dir,
        limit=args.limit,
        offset=args.offset,
    )
    prompt_builder = PromptBuilder.resolve(args.prompt_file)

    saver = OutputSaver(args.output_dir)

    model_config = build_model_config(args)
    model = build_model(args.model, **model_config)

    runner = InferenceRunner(
        model=model,
        prompt_builder=prompt_builder,
        saver=saver,
        chunk_size=args.chunk_size,
    )

    try:
        result = runner.run(dataset)
    finally:
        saver.close()

    metadata.finish()
    metadata.run_id = saver.run_id
    metadata.data_dir = str(args.data_dir.resolve())
    metadata.audio_dir = str(dataset.audios_dir.resolve())
    metadata.metadata_csv = str(dataset.metadata_csv.resolve())
    metadata.output_dir = str(saver.run_dir.resolve())
    metadata.model_name = args.model
    metadata.model_config = model_config
    metadata.prompt_template = prompt_builder.template
    metadata.chunk_size = args.chunk_size
    metadata.num_records_total = len(dataset)
    metadata.num_records_processed = result.num_processed
    metadata.num_records_failed = result.num_failed
    metadata.note = args.note

    metadata_path = saver.save_run_metadata(metadata)

    print(
        f"Processed {result.num_processed}/{len(dataset)} record(s) "
        f"({result.num_failed} failed)."
    )
    print(f"  predictions → {saver.predictions_path.resolve()}")
    print(f"  metadata    → {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
