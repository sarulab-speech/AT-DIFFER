"""Create the 16 kHz audio cache and the train / eval manifests.

The dataset audio is 22.05 kHz, mono, 32-bit float. The Qwen3-Omni feature
extractor takes 16 kHz audio, so the audio is converted once beforehand
instead of being resampled at every epoch. The cache is 16-bit PCM to save
space (the feature extractor converts it back to float).

The manifests are made as for the paper's training:

- train: all rows of the train split. The rows with a difference, then the
  rows without one, each in metadata order, shuffled with random.Random(0).
- eval: the 500 val rows listed in configs/eval_ids.txt, in that order.

Only the audio referenced by the manifests is converted. Files that already
exist in the cache are skipped, so running the command again after an
interruption continues where it stopped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
import struct
import sys
from pathlib import Path

from .data.manifest import Record, write_manifest
from .taxonomy import NO_DIFFERENCE

TARGET_SAMPLE_RATE = 16_000

#: configs/eval_ids.txt, next to src/.
EVAL_IDS_FILE = Path(__file__).resolve().parents[2] / "configs" / "eval_ids.txt"

#: Seed of the shuffle of the train manifest.
TRAIN_SHUFFLE_SEED = 0

#: Maximum number of failed conversions to print.
MAX_FAILURES_SHOWN = 20

csv.field_size_limit(10**9)


def read_metadata(split_dir: Path) -> list[dict[str, str]]:
    """Read ``<split_dir>/metadata.csv`` (written by the dataset stage) as strings."""
    path = split_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_eval_ids(path: Path = EVAL_IDS_FILE) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def order_train_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows of the train manifest, in the order used for the paper."""
    has_difference = [r for r in rows if json.loads(r["difference_types"]) != [NO_DIFFERENCE]]
    no_difference = [r for r in rows if json.loads(r["difference_types"]) == [NO_DIFFERENCE]]
    merged = has_difference + no_difference
    random.Random(TRAIN_SHUFFLE_SEED).shuffle(merged)
    return merged


def select_eval_rows(rows: list[dict[str, str]], eval_ids: list[str]) -> list[dict[str, str]]:
    """The rows whose id is in ``eval_ids``, in the order of ``eval_ids``."""
    by_id = {row["id"]: row for row in rows}
    missing = [i for i in eval_ids if i not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} eval ids not found in the val metadata, e.g. {missing[:5]}")
    return [by_id[i] for i in eval_ids]


def convert_one(source: Path, destination: Path) -> tuple[Path, str | None]:
    """Convert one file to 16 kHz mono 16-bit PCM. Returns (source, status or error).

    The status is "converted" or "skipped" (the destination already exists);
    anything else is an error message.
    """
    if destination.exists():
        return source, "skipped"
    try:
        import librosa
        import soundfile as sf

        # librosa.load returns float32 and resamples to the given rate.
        audio, _ = librosa.load(str(source), sr=TARGET_SAMPLE_RATE, mono=True)
        # Write under a temporary name and rename, so that an interrupted
        # write is never mistaken for a converted file.
        temporary = destination.with_suffix(".tmp.wav")
        sf.write(str(temporary), audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
        temporary.replace(destination)
        return source, "converted"
    except Exception as exc:  # noqa: BLE001 - report every failure at the end
        return source, f"{type(exc).__name__}: {exc}"


def build_cache(pairs: list[tuple[Path, Path]], workers: int, label: str) -> bool:
    """Convert the (source, destination) pairs in parallel. Returns False if any failed."""
    counts = {"converted": 0, "skipped": 0}
    failed: list[tuple[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(convert_one, source, destination) for source, destination in pairs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            source, status = future.result()
            if status in counts:
                counts[status] += 1
            else:
                failed.append((str(source), status))
            if done % 5000 == 0:
                print(f"[prepare] {label}: {done}/{len(pairs)}", flush=True)

    print(
        f"[prepare] {label}: {counts['converted']} converted, "
        f"{counts['skipped']} already present, {len(failed)} failed",
        flush=True,
    )
    for source, error in failed[:MAX_FAILURES_SHOWN]:
        print(f"  {source}: {error}", file=sys.stderr)
    if len(failed) > MAX_FAILURES_SHOWN:
        print(f"  ... and {len(failed) - MAX_FAILURES_SHOWN} more", file=sys.stderr)
    return not failed


def wav_duration_seconds(path: Path) -> float:
    """Duration of a WAV file, read from its RIFF header only."""
    with path.open("rb") as f:
        head = f.read(4096)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")
    pos, block_align, sample_rate = 12, None, None
    while pos + 8 <= len(head):
        chunk_id = head[pos : pos + 4]
        chunk_size = struct.unpack("<I", head[pos + 4 : pos + 8])[0]
        if chunk_id == b"fmt ":
            _, _, sample_rate, _, block_align, _ = struct.unpack(
                "<HHIIHH", head[pos + 8 : pos + 24]
            )
        elif chunk_id == b"data":
            if not block_align or not sample_rate:
                raise ValueError(f"No fmt chunk before the data chunk: {path}")
            return chunk_size / block_align / sample_rate
        pos += 8 + chunk_size + (chunk_size & 1)
    raise ValueError(f"No data chunk in the first 4 KB: {path}")


def cache_path(cache_dir: Path, split: str, row: dict[str, str]) -> Path:
    return cache_dir / split / Path(row["audio_path"]).name


def make_record(row: dict[str, str], audio_path: Path) -> Record:
    return Record(
        id=row["id"],
        audio_path=str(audio_path),
        text=row["text"],
        difference_caption=row["difference_caption"],
        difference_types=json.loads(row["difference_types"]),
        audio_structure=row["audio_structure"],
        duration_seconds=round(wav_duration_seconds(audio_path), 3),
    )


def prepare_split(
    split: str,
    rows: list[dict[str, str]],
    dataset_dir: Path,
    cache_dir: Path,
    manifest_path: Path,
    workers: int,
) -> bool:
    """Convert the audio of ``rows`` and write their manifest. Returns False on failure."""
    split_dir = dataset_dir / split
    (cache_dir / split).mkdir(parents=True, exist_ok=True)
    pairs = [(split_dir / row["audio_path"], cache_path(cache_dir, split, row)) for row in rows]
    if not build_cache(pairs, workers, split):
        return False

    records = [make_record(row, cache_path(cache_dir, split, row)) for row in rows]
    write_manifest(manifest_path, records)
    hours = sum(record.duration_seconds for record in records) / 3600
    print(
        f"[prepare] {split}: {len(records)} records ({hours:.1f} h of audio) -> {manifest_path}",
        flush=True,
    )
    return True


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Output directory of the dataset stage (containing train/ and val/).",
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True,
        help="Directory for the 16 kHz audio cache.",
    )
    parser.add_argument(
        "--manifest-dir", type=Path, required=True,
        help="Directory to write train.jsonl and eval.jsonl to.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel conversion processes (default: 8).",
    )


def run_from_args(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    manifest_dir = args.manifest_dir.expanduser().resolve()

    train_rows = order_train_rows(read_metadata(dataset_dir / "train"))
    eval_rows = select_eval_rows(read_metadata(dataset_dir / "val"), read_eval_ids())

    for split, rows, name in (("train", train_rows, "train.jsonl"), ("val", eval_rows, "eval.jsonl")):
        if not prepare_split(split, rows, dataset_dir, cache_dir, manifest_dir / name, args.workers):
            return 1
    return 0
