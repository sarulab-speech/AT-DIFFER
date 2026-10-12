"""LoRA fine-tuning.

    torchrun --nproc_per_node 2 -m at_differ_finetune.train --config configs/method2.yaml

The loss is chosen by ``loss_reduction`` (loss.build_loss_fn): both
reductions are per-sample normalized, weighted cross-entropy and differ only
in whether the lines are averaged together or normalized separately.
Everything model-specific is in the adapter under models/, so this file does
not depend on the model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .config import Config
from .data.collator import CaptionDataset, ExampleCollator
from .data.manifest import read_manifest
from .data.prompt import PromptBuilder
from .data.targets import line_weight_vector
from .loss import build_loss_fn, concatenate_loss
from .models import build_adapter
from .taxonomy import Taxonomy

logger = logging.getLogger("at_differ_finetune")


def resolve_lora_targets(model: Any, adapter: Any) -> list[str]:
    """List the full names of the modules that LoRA is applied to.

    Passing short names ("q_proj", ...) to peft's target_modules matches them
    by suffix, which would hit both the LLM and the audio encoder. The model
    is scanned here to build a list of **full names** of the LLM's modules,
    leaving out those of the audio encoder.
    """
    suffixes = tuple(adapter.lora_target_modules())
    audio_prefix = adapter.audio_encoder_module_prefix()

    targets, audio_hits = [], 0
    for name, module in model.named_modules():
        if not name.endswith(suffixes):
            continue
        # Skip anything that is not a Linear layer (no other module should
        # have these names, but just in case).
        if not hasattr(module, "weight"):
            continue
        if audio_prefix in name:
            audio_hits += 1
            continue
        targets.append(name)

    if not targets:
        raise RuntimeError(
            f"No module to apply LoRA to (searched suffixes: {suffixes})."
            f" The model structure may have changed; check models/qwen3_omni.py."
        )
    logger.info(
        "LoRA on %d modules (%d modules of the audio encoder excluded)", len(targets), audio_hits
    )
    return targets


def build_trainer_class():
    """Subclass Trainer lazily, so that importing this module does not need transformers."""
    from transformers import Trainer
    from transformers.trainer_pt_utils import LengthGroupedSampler

    #: Inputs used only by the loss. They cannot be passed to the model's forward.
    LOSS_ONLY_INPUTS = ("token_weights", "line_ids")

    class DifferenceCaptionTrainer(Trainer):
        """Trainer with our loss and a length-grouped sampler."""

        def __init__(self, *args, train_lengths=None, loss_fn=concatenate_loss, **kwargs):
            super().__init__(*args, **kwargs)
            self._train_lengths = train_lengths
            self._loss_fn = loss_fn

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss_inputs = {
                key: inputs.pop(key) for key in LOSS_ONLY_INPUTS if key in inputs
            }
            labels = inputs.pop("labels")

            outputs = model(**inputs)
            loss = self._loss_fn(
                outputs.logits,
                labels,
                token_weights=loss_inputs.get("token_weights"),
                line_ids=loss_inputs.get("line_ids"),
            )

            # Put them back for later callbacks and logging.
            inputs["labels"] = labels
            inputs.update(loss_inputs)
            return (loss, outputs) if return_outputs else loss

        def _get_train_sampler(self, *args, **kwargs):
            """Sampler that groups samples by audio duration.

            The text part is nearly the same for every sample, so the padding
            waste is governed by the audio duration (0.5-35 s).
            """
            if not self.args.group_by_length or self._train_lengths is None:
                return super()._get_train_sampler(*args, **kwargs)
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=[int(round(x * 100)) for x in self._train_lengths],
            )

        def _get_eval_sampler(self, eval_dataset=None, *args, **kwargs):
            """Use a plain sequential sampler for evaluation.

            **group_by_length also applies to the eval sampler.** The default
            implementation infers lengths from the dataset and requires its
            items to be dicts with an ``input_ids`` key. CaptionDataset returns
            Examples, so the first evaluation would fail.

            A deterministic order is preferred over length grouping (which uses
            a random permutation), so that eval losses are comparable across
            steps. The eval set has 500 samples, so the padding waste is small.

            The return value follows the default Trainer (None when distributed,
            leaving it to accelerate).
            """
            from torch.utils.data import SequentialSampler

            if eval_dataset is None or not hasattr(eval_dataset, "__len__"):
                return None
            if self.args.world_size <= 1:
                return SequentialSampler(eval_dataset)
            return None

    return DifferenceCaptionTrainer


def load_split(manifest_path: str, adapter: Any, prompt_builder: PromptBuilder) -> CaptionDataset:
    records = read_manifest(manifest_path)
    logger.info("Manifest %s: %d records", manifest_path, len(records))
    return CaptionDataset(records, adapter, prompt_builder)


def run(config: Config) -> None:
    import torch  # noqa: F401  (fail early if torch is missing)
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, set_seed

    set_seed(config.seed)
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the loss before allocating the GPUs, so configuration errors surface here.
    loss_fn = build_loss_fn(config.loss_reduction, line_weight_vector(config.line_weights))
    logger.info(
        "Loss: reduction=%s / supervised_lines=%s / line_weights=%s",
        config.loss_reduction,
        config.supervised_lines,
        config.line_weights or "(all 1.0)",
    )

    taxonomy = Taxonomy.load(config.taxonomy_file, config.output_format_file)
    adapter = build_adapter(
        config.model,
        taxonomy=taxonomy,
        model_path=config.model_path,
        attn_implementation=config.attn_implementation,
        max_audio_seconds=config.max_audio_seconds,
        supervised_lines=config.supervised_lines,
        line_weights=config.line_weights,
    )
    processor = adapter.load_processor()
    prompt_builder = PromptBuilder.from_file(config.prompt_file)

    train_dataset = load_split(config.train_manifest, adapter, prompt_builder)
    eval_dataset = (
        load_split(config.eval_manifest, adapter, prompt_builder)
        if config.eval_manifest
        else None
    )

    # TrainingArguments is created before the model is loaded, in the same
    # order as in the paper's runs.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=config.group_by_length,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        dataloader_num_workers=config.dataloader_num_workers,
        # "The mean of per-sample means equals the mean over all samples" holds
        # only if every micro batch has the same number of samples. Drop the
        # last incomplete batch to make it exact.
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=[],
        seed=config.seed,
    )

    model = adapter.load_model()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolve_lora_targets(model, adapter),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    trainer_class = build_trainer_class()
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ExampleCollator(adapter=adapter, pad_token_id=pad_token_id),
        train_lengths=train_dataset.lengths(),
        loss_fn=loss_fn,
    )

    # Record the settings in the output directory. Only rank 0 writes, since
    # several ranks writing the same file could corrupt it.
    if trainer.is_world_process_zero():
        (output_dir / "train_config.json").write_text(
            json.dumps(vars(config), ensure_ascii=False, indent=2, default=str)
        )

    # Resume from the latest checkpoint-* if there is one, so that a job cut
    # off by the time limit can be continued by running the same command.
    resume = None
    if config.resume_from_checkpoint:
        resume = config.resume_from_checkpoint
    elif any(output_dir.glob("checkpoint-*")):
        resume = True
        logger.info("Found existing checkpoints; resuming: %s", output_dir)

    trainer.train(resume_from_checkpoint=resume)

    if trainer.is_world_process_zero():
        (output_dir / "log_history.json").write_text(
            json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2)
        )

    trainer.save_model(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "adapter"))
    logger.info("Done: %s", output_dir / "adapter")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True,
        help="Training config (YAML), e.g. configs/method2.yaml.",
    )


def run_from_args(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    run(Config.load(args.config))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m at_differ_finetune.train",
        description="LoRA fine-tuning of Qwen3-Omni on AT-DIFFER.",
    )
    add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
