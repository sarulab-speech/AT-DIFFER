"""Reading the bundled metadata and writing it to the output directory."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pandas as pd

# dataset/metadata/, next to src/.
METADATA_DIR = Path(__file__).resolve().parents[2] / "metadata"

SPLITS = ("train", "val", "test")


def metadata_path(split: str) -> Path:
    return METADATA_DIR / f"{split}.csv.gz"


def read_metadata(split: str) -> pd.DataFrame:
    """Read the bundled metadata of ``split``, keeping every value as a string."""
    return pd.read_csv(metadata_path(split), dtype=str, keep_default_na=False)


def write_metadata(split: str, path: Path) -> None:
    """Decompress the bundled metadata of ``split`` to ``path`` byte for byte."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_bytes(gzip.decompress(metadata_path(split).read_bytes()))
    os.replace(tmp_path, path)


def source_clip_ids(metadata: pd.DataFrame) -> list[str]:
    """All FSD50K clip IDs referenced by ``metadata``, without duplicates."""
    ids: dict[str, None] = {}
    for value in metadata["source_clip_ids"]:
        ids.update(dict.fromkeys(json.loads(value)))
    return list(ids)
