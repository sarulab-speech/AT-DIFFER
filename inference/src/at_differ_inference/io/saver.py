from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from at_differ_inference.io.run_metadata import RunMetadata

if TYPE_CHECKING:
    from at_differ_inference.pipeline.runner import PredictionRow

_PREDICTION_COLUMNS = [
    "id",
    "audio_path",
    "audio_structure",
    "structure_params",
    "difference_types",
    "text",
    "difference_caption",
    "model_output",
]


class OutputSaver:
    """Creates a timestamped run directory and writes the two output files.

    ``predictions.csv`` is opened at construction (header written immediately)
    and rows are appended chunk-by-chunk via :pymeth:`append_rows`, so partial
    results survive a crash mid-run. ``run_metadata.yaml`` is written once at the
    end via :pymeth:`save_run_metadata`.

    Output layout::

        {output_root}/
            {YYYYMMDDhhmm}/
                predictions.csv      # one row per record (CSV, written incrementally)
                run_metadata.yaml     # single run's metadata (YAML)
    """

    def __init__(self, output_root: str | Path, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now().strftime("%Y%m%d%H%M")
        self.run_dir = Path(output_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.run_dir / "predictions.csv"
        self.run_metadata_path = self.run_dir / "run_metadata.yaml"

        self._file = self.predictions_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_PREDICTION_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def append_rows(self, rows: list[PredictionRow]) -> None:
        """Append a chunk of rows to predictions.csv and flush to disk."""
        for row in rows:
            self._writer.writerow(
                {
                    "id": row.id,
                    "audio_path": row.audio_path,
                    "audio_structure": row.audio_structure,
                    "structure_params": row.structure_params,
                    "difference_types": row.difference_types,
                    "text": row.text,
                    "difference_caption": row.difference_caption,
                    "model_output": row.model_output,
                }
            )
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def save_run_metadata(self, metadata: RunMetadata) -> Path:
        with self.run_metadata_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                metadata.to_dict(),
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        return self.run_metadata_path

    def __enter__(self) -> "OutputSaver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
