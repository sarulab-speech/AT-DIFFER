"""Reading and writing the training manifests (jsonl).

Training reads a manifest made by ``at-differ-finetune prepare`` rather than
the metadata itself: the order of the records and the paths to the 16 kHz
audio cache are fixed once, outside the training job, and the job uses the
manifest as it is.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Record:
    """One training sample."""

    id: str
    audio_path: str  # absolute path to the 16 kHz cached WAV
    text: str  # text inserted into the prompt
    difference_caption: str  # reference difference caption (the EXPLANATION text)
    difference_types: list[str]  # difference types of the metadata
    audio_structure: str  # single / simultaneous / sequential
    duration_seconds: float  # used for length-grouped batching


def write_manifest(path: str | Path, records: list[Record]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def iter_manifest(path: str | Path) -> Iterator[Record]:
    src = Path(path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Manifest not found: {src}")
    with src.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Record(**json.loads(line))
            except TypeError as exc:
                raise ValueError(f"{src}:{lineno}: unexpected manifest fields: {exc}") from None


def read_manifest(path: str | Path) -> list[Record]:
    return list(iter_manifest(path))
