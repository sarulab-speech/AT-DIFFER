import json
from pathlib import Path

import pytest
import yaml

from at_differ_dataset.metadata import SPLITS, read_metadata

TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "resources" / "taxonomy.yaml"

COLUMNS = [
    "id",
    "audio_path",
    "audio_structure",
    "structure_params",
    "difference_types",
    "text",
    "difference_caption",
    "source_clip_ids",
]

# Section 3.3 of the paper: (total, single, simultaneous, sequential).
EXPECTED_COUNTS = {
    "train": (149_172, 30_770, 51_279, 67_123),
    "val": (14_466, 3_034, 4_941, 6_491),
    "test": (33_642, 6_982, 11_604, 15_056),
}


@pytest.fixture(scope="module")
def taxonomy():
    return yaml.safe_load(TAXONOMY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module", params=SPLITS)
def split_metadata(request):
    return request.param, read_metadata(request.param)


def test_columns(split_metadata):
    _, metadata = split_metadata
    assert list(metadata.columns) == COLUMNS


def test_counts_per_audio_structure(split_metadata):
    split, metadata = split_metadata
    by_structure = metadata["audio_structure"].value_counts().to_dict()
    total, single, simultaneous, sequential = EXPECTED_COUNTS[split]
    assert len(metadata) == total
    assert by_structure == {
        "single": single, "simultaneous": simultaneous, "sequential": sequential,
    }


def test_values_are_in_taxonomy(split_metadata, taxonomy):
    _, metadata = split_metadata
    assert set(metadata["audio_structure"]) <= set(taxonomy["audio_structures"])
    used_types = {t for value in metadata["difference_types"] for t in json.loads(value)}
    assert used_types <= set(taxonomy["difference_types"])


def test_combinations_match_taxonomy(split_metadata, taxonomy):
    _, metadata = split_metadata
    used = {
        (structure, tuple(json.loads(types)))
        for structure, types in zip(metadata["audio_structure"], metadata["difference_types"])
    }
    valid = {
        (structure, tuple(combination))
        for structure, combinations in taxonomy["valid_combinations"].items()
        for combination in combinations
    }
    assert used == valid


def test_ids_and_audio_paths(split_metadata):
    _, metadata = split_metadata
    ids = metadata["id"].astype(int).tolist()
    assert ids == list(range(1, len(metadata) + 1))
    assert metadata["audio_path"].tolist() == [f"audios/audio_{i - 1:05d}.wav" for i in ids]


def test_number_of_source_clips(split_metadata):
    _, metadata = split_metadata
    for structure, value in zip(metadata["audio_structure"], metadata["source_clip_ids"]):
        assert len(json.loads(value)) == (1 if structure == "single" else 2)
