"""Adapters that hide model-specific details.

Models are registered as "module:Class" strings and imported only when
selected, so the heavy dependencies (torch, transformers, peft) are imported
only when an adapter is actually built.
"""

from .registry import available_models, build_adapter

__all__ = ["available_models", "build_adapter"]
