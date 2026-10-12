"""Adapter for Qwen3-Omni-30B-A3B.

The input must be built exactly as in the inference stage; otherwise the
format the model is trained on differs from the one it sees at inference
time. In particular:

  - No system message: a single user turn.
  - The content is [audio, text]: the audio first, then the prompt.
  - The prompt ends with "Caption: {caption}" (filled in by PromptBuilder).

**Non-thinking generation.** An empty think block is inserted right after the
generation prompt, and the model writes the three lines after it. The base
model is a thinking checkpoint, which otherwise produces a long <think> block.
The same prefix is used in inference (model key ``qwen3-omni-nothink``).

Memory
------
The Talker (speech output) is not used, so only the Thinker is loaded
(Qwen3OmniMoeThinkerForConditionalGeneration).

MoE and LoRA
------------
LoRA is applied only to the attention projections (q/k/v/o), not to the
expert FFNs: with 128 experts per layer and only 8 routed per token, adapters
on the experts would each receive very sparse gradients.
"""

from __future__ import annotations

import os
from typing import Any

from ..data.manifest import Record
from ..data.targets import (
    DEFAULT_SUPERVISED_LINES,
    NO_LINE_ID,
    build_supervision,
    build_target,
    encode_line_ids,
    line_of_offsets,
    normalize_supervised_lines,
    split_for_supervision,
)
from ..taxonomy import Taxonomy
from .base import Example, FinetuneAdapter

# Local directory of the base model (see README for how to download it).
DEFAULT_MODEL_PATH = "../outputs/models/Qwen3-Omni-30B-A3B-Thinking"

# Prefix for non-thinking generation: an empty, closed think block. It must be
# exactly the same string as in the inference stage.
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"

# Modules that LoRA is applied to. The MoE expert FFNs are deliberately left
# out (see the module docstring).
ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class Qwen3OmniAdapter(FinetuneAdapter):
    """Adapter for training the Thinker of Qwen3-Omni with LoRA."""

    sample_rate = 16_000

    def __init__(
        self,
        taxonomy: Taxonomy,
        model_path: str = DEFAULT_MODEL_PATH,
        attn_implementation: str = "sdpa",
        max_audio_seconds: float = 40.0,
        supervised_lines=None,
        line_weights: dict[str, float] | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.model_path = os.path.expanduser(model_path)
        self.supervised_lines = normalize_supervised_lines(
            supervised_lines or DEFAULT_SUPERVISED_LINES
        )
        self.line_weights = dict(line_weights or {})
        self.attn_implementation = attn_implementation
        self.max_audio_seconds = max_audio_seconds
        self._processor: Any = None

    # --- Loading ------------------------------------------------------------

    def load_processor(self) -> Any:
        if self._processor is None:
            from transformers import Qwen3OmniMoeProcessor

            self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
            # Right padding for training (nothing is generated).
            self._processor.tokenizer.padding_side = "right"
        return self._processor

    def load_model(self) -> Any:
        import torch
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
        )
        model.config.use_cache = False  # required with gradient checkpointing
        return model

    # --- LoRA ---------------------------------------------------------------

    def lora_target_modules(self) -> list[str]:
        """The attention projections. The audio encoder has modules with the same names.

        peft matches target_modules by suffix, so these names alone would also
        hit the audio encoder. train.resolve_lora_targets excludes the modules
        under audio_encoder_module_prefix().
        """
        return list(ATTENTION_MODULES)

    def audio_encoder_module_prefix(self) -> str:
        """Prefix shared by the module names of the audio encoder (AuT)."""
        return "audio_tower"

    def generation_prefix(self) -> str:
        return EMPTY_THINK_BLOCK

    # --- Building a sample ----------------------------------------------------

    def build_example(self, record: Record, prompt: str) -> Example:
        """Turn one record into input_ids / labels / token_weights.

        The boundary of the target is fixed **by tokenizing the head and the
        body separately and concatenating them**, not by searching for
        "EXPLANATION:" in the tokens, which would shift silently if the
        tokenizer changed.
        """
        import numpy as np
        import soundfile as sf

        processor = self.load_processor()
        target = build_target(record.difference_types, record.difference_caption, self.taxonomy)

        # --- Audio ---
        audio, sample_rate = sf.read(record.audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:  # downmix if a stereo file slipped in
            audio = audio.mean(axis=1)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{record.audio_path} is {sample_rate} Hz. Create the {self.sample_rate} Hz "
                f"cache with `at-differ-finetune prepare` and use its manifests."
            )
        limit = int(self.max_audio_seconds * self.sample_rate)
        if audio.shape[0] > limit:
            audio = audio[:limit]
        audio = np.ascontiguousarray(audio)

        # --- Head (no loss) ---
        # The same message structure as in inference: no system message, audio first.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": record.audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Where the loss starts depends on supervised_lines. With the explanation
        # only (Method 1), head_suffix is target.prefix.
        head_suffix, body_segments = split_for_supervision(target, self.supervised_lines)
        head_text = chat_text + self.generation_prefix() + head_suffix

        head_inputs = processor(
            text=[head_text],
            audio=[audio],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        head_ids = head_inputs["input_ids"][0].tolist()

        # --- Body (loss) ---
        # The body starts with a space (" The caption ..."), so the concatenated
        # tokens match those of the whole text tokenized at once (BPE merges a
        # space with the following word).
        tokenizer = processor.tokenizer
        body_text = "".join(text for _, text in body_segments)
        body_encoding = tokenizer(
            body_text, add_special_tokens=False, return_offsets_mapping=True
        )
        body_ids = list(body_encoding["input_ids"])
        body_offsets = [tuple(pair) for pair in body_encoding["offset_mapping"]]

        # Also teach the model to stop after the sentence.
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or eos_id < 0:
            eos_id = tokenizer.eos_token_id
        body_ids.append(eos_id)
        body_offsets.append((0, 0))  # no text; attributed to the last segment

        # --- Line of each token ---
        # An unsupervised line can lie inside the body. split_for_supervision
        # marks it with None; such tokens get labels of -100 and stay as input.
        body_lines = line_of_offsets(body_offsets, body_segments)

        # Labels (-100 for unsupervised lines) and the line weights.
        body_labels, body_weights = build_supervision(body_ids, body_lines, self.line_weights)

        example = Example(
            input_ids=head_ids + body_ids,
            labels=[-100] * len(head_ids) + body_labels,
            token_weights=[1.0] * len(head_ids) + body_weights,
            # Read only by the separate reduction (Method 3). Built for every
            # Example so that the Example does not depend on the reduction.
            line_ids=[NO_LINE_ID] * len(head_ids) + encode_line_ids(body_lines),
            extras={
                key: value[0]
                for key, value in head_inputs.items()
                if key not in ("input_ids", "attention_mask")
            },
        )
        return example

    # --- Batching -------------------------------------------------------------

    def collate_extras(self, batch: list[Example]) -> dict[str, Any]:
        """Batch the audio features.

        The Qwen3-Omni processor returns variable-length audio as
        ``input_features`` and ``feature_attention_mask``. They are right-padded
        along the time axis (the last dimension) and stacked.
        """
        import torch

        keys = set()
        for example in batch:
            keys.update(example.extras)
        if not keys:
            return {}

        collated: dict[str, Any] = {}
        for key in sorted(keys):
            values = [example.extras[key] for example in batch if key in example.extras]
            if len(values) != len(batch):
                raise ValueError(f"Some samples lack the extras key {key!r}")
            if values[0].dim() == 0:
                collated[key] = torch.stack(values)
                continue
            max_len = max(v.shape[-1] for v in values)
            padded = [
                torch.nn.functional.pad(v, (0, max_len - v.shape[-1])) for v in values
            ]
            collated[key] = torch.stack(padded)
        return collated
