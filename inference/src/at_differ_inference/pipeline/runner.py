from dataclasses import dataclass

from at_differ_inference.data.dataset import InferenceDataset, Record
from at_differ_inference.io.saver import OutputSaver
from at_differ_inference.models.base import AudioCaptioner, AudioTextPair
from at_differ_inference.pipeline.prompt import PromptBuilder


@dataclass(frozen=True)
class PredictionRow:
    """One inference result, ready to be written to predictions.csv."""

    id: str
    audio_path: str
    audio_structure: str
    structure_params: str
    difference_types: str
    text: str
    difference_caption: str
    model_output: str


@dataclass(frozen=True)
class RunResult:
    """Aggregated outcome of a run."""

    num_processed: int
    num_failed: int


class InferenceRunner:
    """Drives chunked inference over a dataset, saving results incrementally.

    For each chunk it builds ``[AudioTextPair, ...]``, calls the model (which
    continuous-batches the chunk internally), pairs the returned strings back
    with their records, and appends them to ``predictions.csv`` via the saver —
    so partial results survive a crash. A chunk that raises is recorded as failed
    (its rows carry an error message) and processing continues with the next.
    """

    def __init__(
        self,
        model: AudioCaptioner,
        prompt_builder: PromptBuilder,
        saver: OutputSaver,
        chunk_size: int = 64,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        self.model = model
        self.prompt_builder = prompt_builder
        self.saver = saver
        self.chunk_size = chunk_size

    def run(self, dataset: InferenceDataset) -> RunResult:
        records = list(dataset)
        total = len(records)
        num_failed = 0
        num_chunks = (total + self.chunk_size - 1) // self.chunk_size

        for start in range(0, total, self.chunk_size):
            chunk = records[start : start + self.chunk_size]
            chunk_num = start // self.chunk_size + 1
            print(
                f"[runner] chunk {chunk_num}/{num_chunks} "
                f"(records {start + 1}-{start + len(chunk)} of {total})",
                flush=True,
            )
            try:
                outputs = self._infer_chunk(chunk)
            except Exception as exc:  # noqa: BLE001 - record and continue
                print(f"[runner] chunk {chunk_num} failed: {exc}", flush=True)
                num_failed += len(chunk)
                rows = self._error_rows(chunk, exc)
            else:
                rows = [
                    self._to_row(record, output)
                    for record, output in zip(chunk, outputs)
                ]
            self.saver.append_rows(rows)

        return RunResult(num_processed=total - num_failed, num_failed=num_failed)

    def _infer_chunk(self, chunk: list[Record]) -> list[str]:
        pairs = [
            AudioTextPair(
                audio_path=record.audio_path,
                text=self.prompt_builder.build(record.text),
            )
            for record in chunk
        ]
        outputs = self.model.generate(pairs)
        if len(outputs) != len(chunk):
            raise ValueError(
                f"Model returned {len(outputs)} result(s) for a chunk of "
                f"{len(chunk)} record(s)."
            )
        return [str(o) for o in outputs]

    @staticmethod
    def _to_row(record: Record, model_output: str) -> PredictionRow:
        return PredictionRow(
            id=record.id,
            audio_path=record.audio_path,
            audio_structure=record.audio_structure,
            structure_params=record.structure_params,
            difference_types=record.difference_types,
            text=record.text,
            difference_caption=record.difference_caption,
            model_output=model_output,
        )

    @classmethod
    def _error_rows(cls, chunk: list[Record], exc: Exception) -> list[PredictionRow]:
        message = f"[ERROR] {type(exc).__name__}: {exc}"
        return [cls._to_row(record, message) for record in chunk]
