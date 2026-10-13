"""Name → AudioCaptioner mapping with lazy imports.

Backends are referenced by ``"module:ClassName"`` strings so that selecting one
model does not import the heavy dependencies of the others (vllm, transformers,
…). The concrete class is imported only when :func:`build_model` is called.
"""

import importlib

from at_differ_inference.models.base import AudioCaptioner

# Registry key → "import.path:ClassName".
MODEL_REGISTRY: dict[str, str] = {
    "qwen3-omni-thinking": "at_differ_inference.models.qwen3_omni:Qwen3OmniCaptioner",
    # Same engine with the reasoning block suppressed. Use it with a LoRA-merged
    # checkpoint from the finetune stage (`at-differ-finetune merge`).
    "qwen3-omni-nothink": "at_differ_inference.models.qwen3_omni:Qwen3OmniNoThinkCaptioner",
}


def registered_models() -> list[str]:
    """Return the list of known registry keys."""
    return sorted(MODEL_REGISTRY)


def build_model(name: str, **config) -> AudioCaptioner:
    """Instantiate the registered captioner ``name`` with ``**config``.

    Args:
        name:   A key of :data:`MODEL_REGISTRY`.
        config: Keyword arguments forwarded to the captioner's constructor.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Registered models: {registered_models()}."
        )

    module_path, class_name = MODEL_REGISTRY[name].split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**config)
