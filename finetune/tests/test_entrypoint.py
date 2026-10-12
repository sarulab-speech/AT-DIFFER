"""The training entry point starts.

``python -m at_differ_finetune.train`` is how torchrun starts training, so it
is run here in a separate process, outside pytest.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

STAGE_DIR = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=STAGE_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_train_module_starts():
    """`python -m at_differ_finetune.train --help` works.

    This imports train.py and builds its argument parser, so a missing import
    at module level or a broken option definition fails here.
    """
    result = _run(["-m", "at_differ_finetune.train", "--help"])
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "--config" in result.stdout


def test_cli_has_all_subcommands():
    result = _run(["-m", "at_differ_finetune.cli", "--help"])
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    for command in ("prepare", "train", "merge"):
        assert command in result.stdout


def test_trainer_overrides_both_samplers():
    """The trainer replaces **both** the train and the eval sampler.

    group_by_length also applies to the eval sampler, whose default
    implementation requires dataset items to be dicts with ``input_ids``.
    CaptionDataset returns Examples, so without the override the first
    evaluation would fail, hours into training.
    """
    pytest.importorskip("transformers", reason="needed to subclass Trainer")
    from at_differ_finetune.train import build_trainer_class

    cls = build_trainer_class()
    for name in ("_get_train_sampler", "_get_eval_sampler", "compute_loss"):
        assert name in cls.__dict__, f"{name} is not overridden"


def test_slurm_scripts_use_a_unique_rendezvous_port():
    """Every torchrun in the Slurm examples gets a port of its own.

    torchrun's default port (29500) is fixed, so a second training job on the
    same node would fail to start (address already in use).
    """
    scripts = sorted((STAGE_DIR / "scripts" / "slurm").glob("*.sh"))
    using_torchrun = [s for s in scripts if re.search(r"^\s*(uv run )?torchrun", s.read_text(), re.M)]
    assert using_torchrun, "no Slurm example starts torchrun"
    offenders = [s.name for s in using_torchrun if not re.search(r"--master[_-]port", s.read_text())]
    assert not offenders, f"{offenders} start torchrun without --master_port"
