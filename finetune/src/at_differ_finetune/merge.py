"""Merge the trained LoRA adapter into the base model and save it as an ordinary checkpoint.

vLLM's LoRA support is unreliable for MoE multimodal models, so the adapter
is merged into the base model instead. The inference stage then only needs
``--model-path <output>``.

The output is the 30B model in bf16, **about 61 GB per checkpoint**.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

#: Sampling parameters. Having them with do_sample=False fails the strict
#: validation of transformers when saving.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def make_generation_config_saveable(generation_config) -> bool:
    """Make the generation config saveable. Returns whether it was changed.

    The generation_config.json of Qwen3-Omni has temperature=0.6, top_p=0.95,
    and top_k=20 but does not set do_sample. transformers only warns about
    this when loading, but **raises an error when saving**, which would stop
    the merged checkpoint from being written.

    Since sampling parameters are given, the intent is do_sample=True, so that
    is set explicitly. The values themselves are kept. Inference does not
    depend on these defaults: the inference stage passes temperature, top_p,
    and top_k to vLLM explicitly.
    """
    if generation_config is None:
        return False
    if getattr(generation_config, "do_sample", False):
        return False  # already consistent
    present = [
        key
        for key in _SAMPLING_KEYS
        if getattr(generation_config, key, None) is not None
    ]
    if not present:
        return False  # without sampling parameters, do_sample=False is correct
    generation_config.do_sample = True
    print(f"[merge] generation_config: set do_sample=True to match {present}")
    return True


def weight_keys(checkpoint_dir: Path) -> set[str]:
    """Weight names of a checkpoint (empty if it has no index)."""
    index = checkpoint_dir / "model.safetensors.index.json"
    if not index.exists():
        return set()
    return set(json.loads(index.read_text())["weight_map"])


def config_identity(checkpoint_dir: Path) -> tuple[str | None, tuple[str, ...]]:
    """(model_type, architectures) of a checkpoint, which vLLM uses to identify the model."""
    raw = json.loads((checkpoint_dir / "config.json").read_text())
    return raw.get("model_type"), tuple(raw.get("architectures") or ())


def verify_output(out_dir: Path, base_path: str | None = None) -> None:
    """Check that the checkpoint was written completely and **has the same structure as the base**.

    Two kinds of failure are detected:

    1. **An incomplete write.** save_pretrained writes the generation config
       before the weights, so a failure at that point leaves only config
       files, which looks like a partial success.
    2. **A structure different from the base.** Saving only the Thinker
       changes model_type and the weight names, and vLLM can then no longer
       identify the model. The merged checkpoint must be a drop-in
       replacement for the base. This check needs the base as a local
       directory.
    """
    shards = sorted(out_dir.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(
            f"No weights were written: {out_dir}\n"
            f"  Contents: {sorted(p.name for p in out_dir.iterdir())}"
        )
    total = sum(p.stat().st_size for p in shards)
    print(f"[merge] Check: {len(shards)} safetensors files, {total / 1024**3:.1f} GiB in total")
    if total < 50 * 1024**3:
        raise RuntimeError(
            f"The weights are too small ({total / 1024**3:.1f} GiB). The 31.7B model in bf16"
            f" should be about 60 GiB. Check whether the write was interrupted."
        )

    if base_path is None:
        return
    base_dir = Path(base_path).expanduser()
    if not (base_dir / "config.json").exists():
        print(f"[merge] Check: no config.json in the base; structure comparison skipped ({base_dir})")
        return

    base_type, base_arch = config_identity(base_dir)
    out_type, out_arch = config_identity(out_dir)
    print(f"[merge] Check: model_type    base={base_type!r} out={out_type!r}")
    print(f"[merge] Check: architectures base={base_arch} out={out_arch}")
    if (out_type, out_arch) != (base_type, base_arch):
        raise RuntimeError(
            "model_type / architectures differ from the base. vLLM would fail to"
            " identify the model with AutoConfig.\n"
            f"  base = {base_type} / {base_arch}\n"
            f"  out  = {out_type} / {out_arch}"
        )

    base_keys, out_keys = weight_keys(base_dir), weight_keys(out_dir)
    if not base_keys or not out_keys:
        print("[merge] Check: no index; weight name comparison skipped")
        return
    missing = base_keys - out_keys
    extra = out_keys - base_keys
    print(f"[merge] Check: weight names base={len(base_keys)} out={len(out_keys)}")
    if missing:
        raise RuntimeError(
            f"{len(missing)} weights of the base are missing, e.g. {sorted(missing)[:5]}"
        )
    if extra:
        # Not necessarily fatal (e.g. an untied lm_head), so only warn, but
        # make sure it is seen: inference may fail to load the checkpoint.
        print(
            f"[merge] WARNING: {len(extra)} weights not in the base, e.g. {sorted(extra)[:5]}\n"
            f"[merge]   If inference fails to load the checkpoint, check these first."
        )
    else:
        print("[merge] Check: weight names identical to the base")


def merge(adapter: Path, out: Path, base_model: str | None = None) -> None:
    import torch
    from peft import PeftModel
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )

    adapter_dir = adapter.expanduser().resolve()
    out_dir = out.expanduser().resolve()

    base_path = base_model
    if base_path is None:
        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        base_path = config["base_model_name_or_path"]
    base_path = str(Path(base_path).expanduser())

    print(f"[merge] base   : {base_path}")
    print(f"[merge] adapter: {adapter_dir}")
    print(f"[merge] out    : {out_dir}")

    # **Load the full model (the wrapper), not only the Thinker.** Training
    # loads only the Thinker, but saving it alone would change the structure
    # of the checkpoint:
    #   model_type  : qwen3_omni_moe -> qwen3_omni_moe_thinker (not in AutoConfig)
    #   weight names: thinker.audio_tower... -> audio_tower... (prefix dropped)
    # vLLM identifies the model with AutoConfig, so the former alone makes
    # loading fail. The output must have the same structure as the base.
    #
    # No Talker is included: this base model has enable_audio_output=False and
    # only thinker.* weights, so transformers does not build the Talker or
    # Code2Wav.
    #
    # The merge runs on the CPU; there is no need to put 61 GB on a GPU.
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="cpu"
    )
    thinker = getattr(model, "thinker", None)
    if thinker is None:
        raise RuntimeError(
            f"The model loaded from {base_path} has no .thinker. The structure of"
            f" the base may have changed; check its config.json."
        )

    # The LoRA weights are named relative to the Thinker (see
    # train.resolve_lora_targets), so they must be applied to the Thinker.
    tie_before = getattr(thinker.config, "tie_word_embeddings", None)
    peft_thinker = PeftModel.from_pretrained(thinker, str(adapter_dir), device_map="cpu")
    model.thinker = peft_thinker.merge_and_unload()

    # After merging, peft unties the embeddings and sets
    # tie_word_embeddings=False. LoRA was applied only to the attention
    # projections, not to the embeddings, so tying is unchanged by the merge.
    # Restore the original value to keep the structure of the base.
    if tie_before is not None:
        model.thinker.config.tie_word_embeddings = tie_before
        if getattr(model.config, "thinker_config", None) is not None:
            model.config.thinker_config.tie_word_embeddings = tie_before

    make_generation_config_saveable(getattr(model, "generation_config", None))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    # The processor files (tokenizer, feature extractor, chat template) are
    # needed by Qwen3OmniMoeProcessor.from_pretrained in inference.
    processor = Qwen3OmniMoeProcessor.from_pretrained(base_path)
    processor.save_pretrained(str(out_dir))

    # Record which adapter the checkpoint was made from.
    for name in ("adapter_config.json", "README.md"):
        source = adapter_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / f"lora_{name}")

    verify_output(out_dir, base_path)

    print(f"[merge] Done: {out_dir}")
    print("[merge] Run inference from the inference stage, e.g.:")
    print(f"  uv run at-differ-inference --model qwen3-omni-nothink --model-path {out_dir} ...")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adapter", type=Path, required=True,
        help="Directory of the trained LoRA adapter (<output_dir>/adapter).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output directory of the merged checkpoint.",
    )
    parser.add_argument(
        "--base-model", default=None,
        help="Directory of the base model (default: the one recorded in adapter_config.json).",
    )


def run_from_args(args: argparse.Namespace) -> int:
    merge(args.adapter, args.out, args.base_model)
    return 0
