from pathlib import Path

import pytest

from at_differ_finetune.taxonomy import Taxonomy

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"


@pytest.fixture(scope="session")
def taxonomy() -> Taxonomy:
    """The taxonomy and output format of resources/, as used in the paper."""
    return Taxonomy.load(RESOURCES_DIR / "taxonomy.yaml", RESOURCES_DIR / "output_format.yaml")
