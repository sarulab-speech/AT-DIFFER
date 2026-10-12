"""Loading the training config (YAML).

The defaults are the values used in the paper's training runs. Relative paths
are relative to the stage directory (finetune/), where the commands are run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .models.qwen3_omni import DEFAULT_MODEL_PATH
from .taxonomy import DEFAULT_OUTPUT_FORMAT_FILE, DEFAULT_TAXONOMY_FILE


@dataclass(kw_only=True)
class Config:
    # --- Inputs ---
    train_manifest: str
    eval_manifest: str | None = None
    prompt_file: str = "../resources/prompt.txt"
    taxonomy_file: str = DEFAULT_TAXONOMY_FILE
    output_format_file: str = DEFAULT_OUTPUT_FORMAT_FILE

    # --- Model ---
    model: str = "qwen3-omni"
    #: Local directory of the base model. Converted to an absolute path by
    #: load(), because it is recorded in adapter_config.json and used as the
    #: default base model by `merge`.
    model_path: str = DEFAULT_MODEL_PATH
    attn_implementation: str = "sdpa"
    #: Audio longer than this is truncated.
    max_audio_seconds: float = 40.0

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # --- Loss ---
    #: Lines that receive a loss: ["explanation"] (Method 1) or
    #: ["discrepancy", "types", "explanation"] (Methods 2 and 3).
    supervised_lines: list[str] = field(default_factory=lambda: ["explanation"])
    #: Loss weight λ of each line. Lines not given get 1.0. Only the ratio
    #: matters. Empty (all 1.0) in the paper's runs.
    line_weights: dict[str, float] = field(default_factory=dict)
    #: How the loss is reduced (loss.LOSS_REDUCTIONS):
    #:   "concatenate" … all supervised tokens averaged together (Methods 1, 2).
    #:   "separate"    … each line normalized, then averaged with λ (Method 3).
    loss_reduction: str = "concatenate"

    # --- Training ---
    output_dir: str
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    seed: int = 0
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    #: LoRA is applied only to attention projections, which are used at every
    #: step, so DDP does not need to search for unused parameters (doing so
    #: costs an extra pass over the autograd graph at every step).
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 4
    #: Checkpoint to resume from. Even if None, training resumes from the
    #: latest checkpoint-* in output_dir if there is one.
    resume_from_checkpoint: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Config not found: {resolved}")
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            # Silently ignoring them would hide typos in setting names.
            raise KeyError(f"Unknown config keys: {sorted(unknown)} (known: {sorted(known)})")
        config = cls(**raw)
        config.model_path = str(Path(config.model_path).expanduser().resolve())
        config.validate()
        return config

    def validate(self) -> None:
        """Check that the settings are consistent, before any job is started."""
        from .data.targets import LINE_NAMES, normalize_supervised_lines

        # Unknown line names and a missing explanation are rejected here.
        self.supervised_lines = list(normalize_supervised_lines(self.supervised_lines))

        unknown = sorted(set(self.line_weights) - set(LINE_NAMES))
        if unknown:
            raise ValueError(
                f"Unknown line names in line_weights: {unknown}. Valid names: {list(LINE_NAMES)}."
            )
        for name, weight in self.line_weights.items():
            if weight <= 0:
                raise ValueError(f"line_weights[{name!r}] = {weight}. It must be positive.")
        ignored = sorted(set(self.line_weights) - set(self.supervised_lines))
        if ignored:
            # A weight on an unsupervised line has no effect; reject it rather than ignore it.
            raise ValueError(
                f"line_weights has {ignored}, which are not in supervised_lines"
                f" ({self.supervised_lines})."
            )

        from .loss import LOSS_REDUCTIONS, SEPARATE

        if self.loss_reduction not in LOSS_REDUCTIONS:
            raise ValueError(
                f"Unknown loss_reduction: {self.loss_reduction!r}."
                f" Valid values: {list(LOSS_REDUCTIONS)}."
            )
        if self.loss_reduction == SEPARATE and len(self.supervised_lines) < 2:
            # With one line, separate is identical to concatenate, so the
            # setting (usually a forgotten supervised_lines) would have no effect.
            raise ValueError(
                f"loss_reduction={SEPARATE!r} but supervised_lines is"
                f" {self.supervised_lines}, a single line. Normalizing per line only"
                f" makes sense when two or more lines are supervised."
            )
