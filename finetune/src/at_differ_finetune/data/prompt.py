"""Loading the prompt template and filling it in.

This behaves the same as the inference stage. The placeholder is replaced
with **str.replace, not str.format**, so that other braces in the prompt
cannot break it.
"""

from __future__ import annotations

from pathlib import Path

CAPTION_PLACEHOLDER = "{caption}"


class PromptBuilder:
    """Replace {caption} in the template with the text of a sample."""

    def __init__(self, template: str) -> None:
        if CAPTION_PLACEHOLDER not in template:
            raise ValueError(
                f"The prompt has no {CAPTION_PLACEHOLDER}, so there is no place to insert the text."
            )
        self.template = template

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptBuilder":
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Prompt file not found: {resolved}")
        return cls(resolved.read_text(encoding="utf-8"))

    def build(self, caption: str) -> str:
        return self.template.replace(CAPTION_PLACEHOLDER, caption)
