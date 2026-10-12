"""Common interface of the model adapters.

The training loop (train.py) and the loss (loss.py) do not depend on the
model. Everything model-specific is in an adapter, which is responsible for:
  1. loading the model and the processor (which submodules, dtype, attention);
  2. turning one sample into input_ids / labels / audio features (build_example);
  3. naming the modules that LoRA is applied to;
  4. giving the prefix for non-thinking generation (which must match inference).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..data.manifest import Record
from ..data.targets import NO_LINE_ID


@dataclass
class Example:
    """One sample in the form the model takes.

    input_ids, labels, token_weights, and line_ids have the same length.
    labels are -100 except at supervised positions. token_weights are the
    loss weights (the line weights; see data.targets.build_supervision); the
    weights at positions whose label is -100 are not used.

    line_ids give the line (0 = none) each token belongs to, and are read only
    by the separate reduction (Method 3). **The same Example serves both
    reductions**: the supervised range and the weights do not depend on the
    reduction (see the loss module for how λ takes effect).

    extras holds model-specific inputs (audio features, their mask, ...). The
    collator does not interpret them; it pads and stacks each key.
    """

    input_ids: list[int]
    labels: list[int]
    token_weights: list[float]
    #: If omitted, every token is "no line". Not read by the concatenate reduction.
    line_ids: list[int] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.line_ids:
            self.line_ids = [NO_LINE_ID] * len(self.input_ids)
        lengths = (
            len(self.input_ids),
            len(self.labels),
            len(self.token_weights),
            len(self.line_ids),
        )
        if len(set(lengths)) != 1:
            raise ValueError(
                "input_ids / labels / token_weights / line_ids have different lengths: "
                + " / ".join(str(n) for n in lengths)
            )


class FinetuneAdapter(ABC):
    """Base class for the model-specific processing."""

    #: Sample rate of the audio (the audio cache is made at this rate).
    sample_rate: int = 16_000

    @abstractmethod
    def load_model(self) -> Any:
        """Return the model to train (before LoRA is applied)."""

    @abstractmethod
    def load_processor(self) -> Any:
        """Return the processor that handles text and audio."""

    @abstractmethod
    def build_example(self, record: Record, prompt: str) -> Example:
        """Turn one record into an Example.

        ``prompt`` is the complete prompt, with {caption} already replaced.
        """

    @abstractmethod
    def lora_target_modules(self) -> list[str]:
        """Names of the modules that LoRA is applied to."""

    @abstractmethod
    def generation_prefix(self) -> str:
        """String inserted right after the generation prompt of the chat template.

        Training and inference are non-thinking, so for thinking models this
        is an empty think block. It must be the same string as in inference;
        otherwise the model is trained to start its output differently from
        how it is prompted at inference time.
        """

    def collate_extras(self, batch: list[Example]) -> dict[str, Any]:
        """Batch the extras. They are model-specific, so the adapter implements this."""
        return {}
