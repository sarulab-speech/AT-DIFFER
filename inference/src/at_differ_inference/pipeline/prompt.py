from pathlib import Path

# The prompt shared with the finetune stage. Relative to the stage directory
# (inference/), where the command is run.
DEFAULT_PROMPT_FILE = Path("../resources/prompt.txt")
_PLACEHOLDER = "{caption}"


class PromptBuilder:
    """Wraps a caption in an instruction prompt fed to the model as ``text``.

    The template must contain the ``{caption}`` placeholder, which is replaced
    with each record's caption. The resolved template string is exposed via
    :pyattr:`template` so it can be recorded in the run metadata.
    """

    def __init__(self, template: str) -> None:
        if _PLACEHOLDER not in template:
            raise ValueError(
                f"Prompt template must contain the {_PLACEHOLDER!r} placeholder."
            )
        self.template = template

    @classmethod
    def resolve(cls, template_file: str | Path = DEFAULT_PROMPT_FILE) -> "PromptBuilder":
        """Load the template from ``template_file`` (by default ``../resources/prompt.txt``)."""
        path = Path(template_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return cls(path.read_text(encoding="utf-8"))

    def build(self, caption: str) -> str:
        """Return the prompt text for a given caption."""
        return self.template.replace(_PLACEHOLDER, caption)
