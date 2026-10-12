"""Command-line entry point: ``at-differ-finetune {prepare,train,merge}``."""

from __future__ import annotations

import argparse

from . import merge, prepare, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="at-differ-finetune",
        description="LoRA fine-tuning of Qwen3-Omni on AT-DIFFER (Methods 1-3 of the paper).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = {
        "prepare": (prepare, "Create the 16 kHz audio cache and the train / eval manifests."),
        "train": (train, "Train the LoRA adapter in a single process (use torchrun for several GPUs)."),
        "merge": (merge, "Merge the adapter into the base model for the inference stage."),
    }
    for name, (module, help_text) in commands.items():
        module.add_arguments(subparsers.add_parser(name, help=help_text, description=help_text))

    args = parser.parse_args(argv)
    module, _ = commands[args.command]
    return module.run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
