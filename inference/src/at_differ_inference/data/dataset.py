import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    """A single inference target loaded from metadata.csv.

    Attributes:
        id:                 Unique record ID (the ``id`` column of metadata.csv).
        audio_path:         Absolute path to the WAV file, resolved against data_dir.
        text:               The text fed to the model (``text`` column).
        difference_caption: The ground-truth difference caption (``difference_caption``
                            column), kept for later comparison/evaluation.
        audio_structure:    Structure used to build the audio (``audio_structure``).
        structure_params:   JSON params of that structure (``structure_params``).
        difference_types:   JSON list of the difference types (``difference_types``).
    """

    id: str
    audio_path: str
    text: str
    difference_caption: str
    audio_structure: str = ""
    structure_params: str = ""
    difference_types: str = ""


class InferenceDataset:
    """Loads inference targets from a ``{data_dir}/metadata.csv`` + ``audios/`` layout.

    Expects:
    - ``{data_dir}/metadata.csv`` with at least the columns ``id``,
      ``audio_path``, ``text`` and ``difference_caption``.
    - ``{data_dir}/audios/`` containing the referenced WAV files. The
      ``audio_path`` column holds paths relative to ``data_dir`` (e.g.
      ``audios/audio_00000.wav``); these are resolved to absolute paths.
    """

    ID_COLUMN = "id"
    AUDIO_COLUMN = "audio_path"
    TEXT_COLUMN = "text"
    REFERENCE_COLUMN = "difference_caption"
    METADATA_FILENAME = "metadata.csv"
    AUDIOS_DIRNAME = "audios"
    # Optional metadata columns carried through to predictions.csv (empty if absent).
    AUDIO_STRUCTURE_COLUMN = "audio_structure"
    STRUCTURE_PARAMS_COLUMN = "structure_params"
    DIFFERENCE_TYPES_COLUMN = "difference_types"

    def __init__(
        self,
        data_dir: str | Path,
        limit: int | None = None,
        offset: int = 0,
    ) -> None:
        """Load and validate records.

        Args:
            data_dir:       Directory containing metadata.csv and audios/.
            limit:          If given, keep only N records (after applying offset).
            offset:         Skip the first ``offset`` records before loading. Combined
                            with ``limit``, selects the slice [offset, offset+limit).
        """
        self.data_dir = Path(data_dir)
        self._metadata_csv = self.data_dir / self.METADATA_FILENAME
        self._audios_dir = self.data_dir / self.AUDIOS_DIRNAME

        if not self._metadata_csv.exists():
            raise FileNotFoundError(f"metadata.csv not found: {self._metadata_csv}")
        if not self._audios_dir.is_dir():
            raise FileNotFoundError(f"audios/ directory not found: {self._audios_dir}")

        self._records = self._load(limit, offset)

    @property
    def metadata_csv(self) -> Path:
        return self._metadata_csv

    @property
    def audios_dir(self) -> Path:
        return self._audios_dir

    def _load(self, limit: int | None, offset: int) -> list[Record]:
        with self._metadata_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._require_columns(reader.fieldnames)
            records: list[Record] = []
            for idx, (line_no, row) in enumerate(
                enumerate(reader, start=2)  # line 1 is the header
            ):
                if idx < offset:  # skip the first `offset` data rows
                    continue
                records.append(self._build_record(row, line_no))
                if limit is not None and len(records) >= limit:
                    break
        if not records:
            raise ValueError(
                f"No records found in {self._metadata_csv} "
                f"(offset={offset}, limit={limit})."
            )
        return records

    def _require_columns(self, fieldnames: list[str] | None) -> None:
        required = {
            self.ID_COLUMN,
            self.AUDIO_COLUMN,
            self.TEXT_COLUMN,
            self.REFERENCE_COLUMN,
        }
        present = set(fieldnames or [])
        missing = required - present
        if missing:
            raise ValueError(
                f"metadata.csv is missing required column(s): {sorted(missing)}. "
                f"Found columns: {sorted(present)}."
            )

    def _build_record(self, row: dict[str, str], line_no: int) -> Record:
        record_id = (row.get(self.ID_COLUMN) or "").strip()
        if not record_id:
            raise ValueError(f"Empty 'id' at metadata.csv line {line_no}.")

        rel_audio = (row.get(self.AUDIO_COLUMN) or "").strip()
        audio_path = (self.data_dir / rel_audio).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found for id={record_id!r} (line {line_no}): {audio_path}"
            )

        return Record(
            id=record_id,
            audio_path=str(audio_path),
            text=row.get(self.TEXT_COLUMN) or "",
            difference_caption=row.get(self.REFERENCE_COLUMN) or "",
            audio_structure=row.get(self.AUDIO_STRUCTURE_COLUMN) or "",
            structure_params=row.get(self.STRUCTURE_PARAMS_COLUMN) or "",
            difference_types=row.get(self.DIFFERENCE_TYPES_COLUMN) or "",
        )

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)
