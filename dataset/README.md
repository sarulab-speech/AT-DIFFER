# AT-DIFFER dataset

This directory re-renders the AT-DIFFER dataset from the metadata bundled here
(`metadata/`) and the audio of [FSD50K](https://zenodo.org/records/4060432).
The result is identical to the dataset used in the paper: the same audio–text
pairs, difference captions, and labels for the train, val, and test splits.

Only FSD50K audio is needed as input. The bundled metadata records, for every
item, which FSD50K clips are used and how they are combined, so no random
choices are made during re-rendering.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.11 and the pinned
  dependencies into `dataset/.venv`.
- FSD50K audio from [Zenodo](https://zenodo.org/records/4060432):
  - `FSD50K.dev_audio` (used for train and val)
  - `FSD50K.eval_audio` (used for test)

  Both are distributed as split zip archives. Merge the parts and extract
  them, for example:

  ```bash
  zip -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio_unsplit.zip
  unzip FSD50K.dev_audio_unsplit.zip
  zip -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio_unsplit.zip
  unzip FSD50K.eval_audio_unsplit.zip
  ```

  Each directory should contain `<clip ID>.wav` files. The FSD50K label files
  and FSD50K-Solo are not needed.

## Usage

```bash
cd dataset
uv sync
uv run at-differ-dataset \
    --dev-audio-dir  /path/to/FSD50K.dev_audio \
    --eval-audio-dir /path/to/FSD50K.eval_audio \
    --output-dir     /path/to/AT-DIFFER
```

| Argument | Description |
|---|---|
| `--dev-audio-dir` | `FSD50K.dev_audio` directory. Required when creating train or val. |
| `--eval-audio-dir` | `FSD50K.eval_audio` directory. Required when creating test. |
| `--output-dir` | Output directory. A directory is created under it for each split. |
| `--splits` | Splits to create (default: `train val test`). |

Before writing anything for a split, the command checks that every FSD50K clip
it needs exists. If any are missing, it lists them and exits with a non-zero
status.

**Resuming.** Existing audio files are skipped, so running the same command
again after an interruption creates only the missing files. Each file is
written under a temporary name and renamed when complete, so a file with its
final name is never partial.

## Output

```
<output-dir>/
├── train/
│   ├── metadata.csv
│   └── audios/audio_00000.wav, audio_00001.wav, ...
├── val/
└── test/
```

`metadata.csv` is the bundled `metadata/<split>.csv.gz`, decompressed. The audio
is mono 32-bit float WAV at 22.05 kHz.

| Split | Items | single | simultaneous | sequential | Audio size |
|---|---:|---:|---:|---:|---:|
| train | 149,172 | 30,770 | 51,279 | 67,123 | 87.4 GB |
| val | 14,466 | 3,034 | 4,941 | 6,491 | 10.7 GB |
| test | 33,642 | 6,982 | 11,604 | 15,056 | 27.8 GB |
| total | 197,280 | 40,786 | 67,824 | 88,670 | 125.9 GB |

## Metadata columns

| Column | Description |
|---|---|
| `id` | Serial number starting from 1. |
| `audio_path` | Path of the audio file relative to the split directory: `audios/audio_{id-1:05d}.wav`. |
| `audio_structure` | `single`, `simultaneous`, or `sequential`. |
| `structure_params` | JSON object. `{"gap_seconds": 0.5}` for `sequential` (silence between the two sounds), `{}` otherwise. |
| `difference_types` | JSON array of the difference types between the audio and the text, e.g. `["semantic_difference", "sequential_discrepancy"]`. `["no_difference"]` when the text describes the audio correctly. |
| `text` | Text paired with the audio. |
| `difference_caption` | Reference difference caption (one sentence). |
| `source_clip_ids` | JSON array of the FSD50K clip IDs (`fname`) used for the audio, in the order they are combined: one for `single`, two otherwise. |

The values of `audio_structure` and `difference_types`, and the combinations
that occur in the dataset, are defined in
[`../resources/taxonomy.yaml`](../resources/taxonomy.yaml).

## How the audio is rendered

Each source clip is read as mono float32 and resampled to 22.05 kHz with soxr
(`soxr_hq`). Then, depending on `audio_structure`:

- **single**: the clip as it is.
- **sequential**: the first clip, `gap_seconds` of silence, then the second clip.
- **simultaneous**: leading and trailing silence is trimmed from each clip,
  both are cut to the shorter length, the second clip is scaled to the RMS of
  the first (gain limited to 0.1–10×), and the sum is peak-normalized.

The implementation is in `src/at_differ_dataset/render.py`.

## Reproducibility

On macOS (arm64) with the pinned versions in `pyproject.toml` (Python 3.11,
numpy 1.26.4, soundfile 0.14.0, soxr 1.1.0), all 197,280 re-rendered audio
files were verified to be sample-for-sample identical to the dataset used in
the paper. On other platforms, small numerical differences may occur.

The WAV files are not byte-identical to ours even when the audio matches:
libsndfile writes the time of writing into the `PEAK` chunk of the header.

## License

The re-rendered audio is derived from FSD50K clips and is subject to the
Creative Commons license of each source clip. Per-clip licenses are listed in
`FSD50K.metadata/dev_clips_info_FSD50K.json` and
`FSD50K.metadata/eval_clips_info_FSD50K.json`, distributed with FSD50K; they
include CC0, CC-BY, CC-BY-NC, and CC Sampling+. FSD50K as a whole is licensed
under CC-BY 4.0.

## Tests

```bash
cd dataset
uv run pytest
```

The tests use synthetic signals and the bundled metadata only; FSD50K is not
needed.
