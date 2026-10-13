import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class RunMetadata:
    """Metadata describing a single inference run, saved to run_metadata.yaml.

    Timing fields are filled by :pymeth:`start` / :pymeth:`finish`; the rest are
    set by the caller once the run is configured.
    """

    run_id: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    data_dir: str = ""
    audio_dir: str = ""
    metadata_csv: str = ""
    output_dir: str = ""

    model_name: str = ""
    model_config: dict = field(default_factory=dict)

    prompt_template: str = ""
    chunk_size: int = 0

    num_records_total: int = 0
    num_records_processed: int = 0
    num_records_failed: int = 0

    note: str = ""
    command: str = field(default_factory=lambda: " ".join(sys.argv))
    python_version: str = field(default_factory=lambda: sys.version.split()[0])

    _start_dt: datetime | None = field(default=None, repr=False, compare=False)

    def start(self) -> None:
        self._start_dt = datetime.now()
        self.start_time = self._start_dt.isoformat(timespec="seconds")

    def finish(self) -> None:
        end_dt = datetime.now()
        self.end_time = end_dt.isoformat(timespec="seconds")
        if self._start_dt is not None:
            self.duration_seconds = round((end_dt - self._start_dt).total_seconds(), 3)

    def to_dict(self) -> dict:
        """Serializable view (drops private timing helper)."""
        data = asdict(self)
        data.pop("_start_dt", None)
        return data
