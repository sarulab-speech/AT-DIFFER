"""Command-line entry point: re-render AT-DIFFER from the bundled metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from .audio import load_clip, write_wav
from .metadata import SPLITS, read_metadata, source_clip_ids, write_metadata
from .render import TARGET_SAMPLE_RATE, render

# FSD50K audio directory used by each split.
SPLIT_AUDIO_DIR_ARG = {"train": "dev_audio_dir", "val": "dev_audio_dir", "test": "eval_audio_dir"}

# Maximum number of missing clip IDs to print.
MAX_MISSING_SHOWN = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="at-differ-dataset",
        description="Re-render the AT-DIFFER dataset from the bundled metadata and FSD50K audio.",
    )
    parser.add_argument(
        "--dev-audio-dir", type=Path,
        help="FSD50K.dev_audio directory (required for train and val).",
    )
    parser.add_argument(
        "--eval-audio-dir", type=Path,
        help="FSD50K.eval_audio directory (required for test).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory. A directory is created under it for each split.",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
        help="Splits to create (default: all).",
    )
    args = parser.parse_args(argv)
    for split in args.splits:
        name = SPLIT_AUDIO_DIR_ARG[split]
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required to create {split}.")
    return args


def build_split(split: str, audio_dir: Path, output_dir: Path) -> bool:
    """Create one split. Returns False, without writing anything, if any
    source clip is missing."""
    metadata = read_metadata(split)

    missing = [i for i in source_clip_ids(metadata) if not (audio_dir / f"{i}.wav").is_file()]
    if missing:
        print(
            f"[{split}] {len(missing)} source clip(s) not found in {audio_dir}:",
            file=sys.stderr,
        )
        for clip_id in missing[:MAX_MISSING_SHOWN]:
            print(f"  {clip_id}.wav", file=sys.stderr)
        if len(missing) > MAX_MISSING_SHOWN:
            print(f"  ... and {len(missing) - MAX_MISSING_SHOWN} more", file=sys.stderr)
        return False

    split_dir = output_dir / split
    (split_dir / "audios").mkdir(parents=True, exist_ok=True)
    write_metadata(split, split_dir / "metadata.csv")

    n_skipped = 0
    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc=split, unit="clip"):
        path = split_dir / row.audio_path
        if path.exists():
            n_skipped += 1
            continue
        clips = [load_clip(audio_dir / f"{i}.wav") for i in json.loads(row.source_clip_ids)]
        audio = render(row.audio_structure, json.loads(row.structure_params), clips)
        write_wav(path, audio, TARGET_SAMPLE_RATE)

    print(
        f"[{split}] {len(metadata) - n_skipped} rendered, {n_skipped} already present "
        f"-> {split_dir}"
    )
    return True


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for split in args.splits:
        audio_dir = getattr(args, SPLIT_AUDIO_DIR_ARG[split])
        if not build_split(split, audio_dir, args.output_dir):
            sys.exit(1)


if __name__ == "__main__":
    main()
