"""Dataset and collator.

The dataset only passes each manifest record to the adapter to make an
Example. Padding is done per batch by the collator; padding to a fixed length
would turn the spread of audio durations (0.5-35 s) into wasted computation.

Padding values:
  input_ids      -> pad_token_id (attention_mask is 0, so the value does not matter)
  labels         -> -100 (excluded from the loss)
  token_weights  -> 0.0 (unused since the labels are -100; 0 makes a bug in
                         the loss show up as a harmless zero weight)
  line_ids       -> NO_LINE_ID (= 0, "no line"; matches labels of -100)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models.base import Example, FinetuneAdapter
from .manifest import Record
from .prompt import PromptBuilder
from .targets import NO_LINE_ID

LABEL_IGNORE_INDEX = -100


class CaptionDataset:
    """Dataset that turns each manifest record into an Example through the adapter.

    It does not subclass torch.utils.data.Dataset, so that it can be built
    and tested without importing torch. It has __len__ and __getitem__, so
    DataLoader treats it as an ordinary map-style dataset.
    """

    def __init__(
        self,
        records: list[Record],
        adapter: FinetuneAdapter,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.records = records
        self.adapter = adapter
        self.prompt_builder = prompt_builder

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Example:
        record = self.records[index]
        return self.adapter.build_example(record, self.prompt_builder.build(record.text))

    def lengths(self) -> list[float]:
        """Length measure for length-grouped batching.

        The text part is nearly the same for every sample (the prompt is about
        2,300 tokens), so the amount of padding is governed by the audio
        duration. Counting tokens would require preprocessing every sample, so
        the duration is used directly as the measure.
        """
        return [record.duration_seconds for record in self.records]


@dataclass
class ExampleCollator:
    """Turn a list of Examples into a batch dict for the model."""

    adapter: FinetuneAdapter
    pad_token_id: int

    def __call__(self, batch: list[Example]) -> dict[str, Any]:
        import torch

        max_len = max(len(example.input_ids) for example in batch)

        input_ids, attention_mask, labels, token_weights, line_ids = [], [], [], [], []
        for example in batch:
            pad = max_len - len(example.input_ids)
            input_ids.append(example.input_ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(example.input_ids) + [0] * pad)
            labels.append(example.labels + [LABEL_IGNORE_INDEX] * pad)
            token_weights.append(example.token_weights + [0.0] * pad)
            line_ids.append(example.line_ids + [NO_LINE_ID] * pad)

        collated: dict[str, Any] = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": torch.tensor(token_weights, dtype=torch.float32),
            "line_ids": torch.tensor(line_ids, dtype=torch.long),
        }
        collated.update(self.adapter.collate_extras(batch))
        return collated
