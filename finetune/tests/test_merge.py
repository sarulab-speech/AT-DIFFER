"""The parts of merge that can be checked without loading the model.

The generation config of the base model has temperature / top_p / top_k
without do_sample. transformers accepts this when loading but rejects it when
saving, and it fails **before the weights are written**, leaving an output
directory with only config files that looks like a partial success.
"""

import json
import types
from pathlib import Path

import pytest

from at_differ_finetune.merge import make_generation_config_saveable, verify_output


def _generation_config(**kwargs):
    """Stand-in for GenerationConfig (only attribute access is used)."""
    return types.SimpleNamespace(**kwargs)


def test_sampling_params_without_do_sample_are_reconciled():
    """Set do_sample so that the config can be saved."""
    gen = _generation_config(do_sample=False, temperature=0.6, top_p=0.95, top_k=20)
    assert make_generation_config_saveable(gen) is True
    assert gen.do_sample is True
    # The values of the base are kept.
    assert (gen.temperature, gen.top_p, gen.top_k) == (0.6, 0.95, 20)


def test_already_consistent_config_is_untouched():
    gen = _generation_config(do_sample=True, temperature=0.6)
    assert make_generation_config_saveable(gen) is False
    assert gen.do_sample is True


def test_greedy_config_stays_greedy():
    """Without sampling parameters, do_sample=False is correct and is left alone."""
    gen = _generation_config(do_sample=False, temperature=None, top_p=None, top_k=None)
    assert make_generation_config_saveable(gen) is False
    assert gen.do_sample is False


def test_missing_generation_config_is_not_an_error():
    assert make_generation_config_saveable(None) is False


def test_verify_output_rejects_a_directory_without_weights(tmp_path):
    """An output directory with only config files is not taken as a success."""
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="No weights were written"):
        verify_output(tmp_path)


def test_verify_output_rejects_a_truncated_write(tmp_path):
    (tmp_path / "model-00001-of-00016.safetensors").write_bytes(b"x" * 1024)
    with pytest.raises(RuntimeError, match="too small"):
        verify_output(tmp_path)


def _make_checkpoint(path: Path, model_type: str, arch: str, keys: list[str]):
    """A checkpoint with config.json, an index, and an empty safetensors file."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": model_type, "architectures": [arch]}))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}})
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"")


def _fill(path: Path, size: int):
    """Give a file an apparent size with a sparse file (no disk space is used)."""
    with open(path, "wb") as f:
        f.truncate(size)
    assert path.stat().st_size == size


#: Size that passes the 50 GiB check of verify_output.
_BIG = 51 * 1024**3


def test_verify_detects_a_thinker_only_export(tmp_path):
    """Saving only the Thinker changes model_type, which must be detected."""
    base = tmp_path / "base"
    out = tmp_path / "out"
    _make_checkpoint(base, "qwen3_omni_moe", "Qwen3OmniMoeForConditionalGeneration",
                     ["thinker.audio_tower.conv2d1.bias"])
    _make_checkpoint(out, "qwen3_omni_moe_thinker", "Qwen3OmniMoeThinkerForConditionalGeneration",
                     ["audio_tower.conv2d1.bias"])
    _fill(out / "model-00001-of-00001.safetensors", _BIG)

    with pytest.raises(RuntimeError, match="model_type / architectures differ"):
        verify_output(out, str(base))


def test_verify_detects_missing_weights(tmp_path):
    """Even with the right model_type, missing weights are detected."""
    base = tmp_path / "base"
    out = tmp_path / "out"
    _make_checkpoint(base, "qwen3_omni_moe", "A", ["thinker.a", "thinker.b"])
    _make_checkpoint(out, "qwen3_omni_moe", "A", ["thinker.a"])
    _fill(out / "model-00001-of-00001.safetensors", _BIG)

    with pytest.raises(RuntimeError, match="missing"):
        verify_output(out, str(base))


def test_verify_passes_for_a_faithful_export(tmp_path, capsys):
    """An output with the same structure as the base passes."""
    base = tmp_path / "base"
    out = tmp_path / "out"
    keys = ["thinker.audio_tower.conv2d1.bias", "thinker.model.layers.0.self_attn.q_proj.weight"]
    _make_checkpoint(base, "qwen3_omni_moe", "Qwen3OmniMoeForConditionalGeneration", keys)
    _make_checkpoint(out, "qwen3_omni_moe", "Qwen3OmniMoeForConditionalGeneration", keys)
    _fill(out / "model-00001-of-00001.safetensors", _BIG)

    verify_output(out, str(base))
    assert "weight names identical to the base" in capsys.readouterr().out
