from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioTextPair:
    """One inference input: an audio file plus the final prompt text.

    Attributes:
        audio_path: Absolute path to the audio (WAV) file.
        text:       The prompt text already built from the caption (i.e. the
                    PromptBuilder output), fed to the model as-is.
    """

    audio_path: str
    text: str


class AudioCaptioner(ABC):
    """Abstract interface for an audio+text captioning model.

    A concrete subclass owns its (heavy) engine, constructed once and reused
    across all ``generate`` calls. Adding a new backend means implementing this
    interface and registering the class in ``at_differ_inference.models.registry``.

    Subclasses should set :pyattr:`name` to a short label (recorded in run
    metadata).
    """

    name: str = "audio-captioner"

    @abstractmethod
    def generate(self, pairs: list[AudioTextPair]) -> list[str]:
        """Run inference on a batch of pairs.

        The returned list must be the same length as ``pairs`` and aligned by
        index: ``result[i]`` is the model's output text for ``pairs[i]``.
        """
        raise NotImplementedError
