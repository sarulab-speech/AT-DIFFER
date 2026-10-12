"""Mapping from model names to adapter implementations (imported lazily).

Values are "module:Class" strings, imported only when selected, so that
building a manifest or running the tests does not import torch or
transformers.

To add a model, implement base.FinetuneAdapter and add one line to _REGISTRY.
"""

from __future__ import annotations

import importlib
from typing import Any

_REGISTRY: dict[str, str] = {
    "qwen3-omni": "at_differ_finetune.models.qwen3_omni:Qwen3OmniAdapter",
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_adapter(name: str, **kwargs: Any):
    """Build an adapter by name."""
    try:
        spec = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Available: {available_models()}."
        ) from None
    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)
