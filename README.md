# wispr

`wispr` is a Python library and CLI for generating synchronized, standard line-level
`.lrc` files from full-song audio plus a canonical line-by-line lyrics file.

The lyrics file is the source of truth. Transcription and alignment backends exist to
recover timing, not to rewrite lyrics.

## Demo contract

```bash
wispr song.wav lyrics.txt
```

By default this writes `song.lrc` next to the audio file. Existing outputs are not
overwritten unless `--force` is passed.

```bash
wispr song.wav lyrics.txt -o output.lrc --debug --force
```

Initial flags:

- `-o, --output`: choose an output path
- `--force`: overwrite an existing output file
- `--debug`: write `transcript.json`, `alignment.json`, and `segments.json`
- `--backend mock|whisperx`: choose mocked timing or the optional WhisperX backend
- `--demucs`: run optional Demucs vocal separation before WhisperX
- `--model`: WhisperX model name, defaulting to `base`
- `--device`: runtime device, defaulting to `auto`
- `--compute-type`: WhisperX compute type, defaulting to `auto`
- `--batch-size`: WhisperX transcription batch size, defaulting by device
- `--language`: WhisperX language code, defaulting to `en`

## Current milestone

The default backend uses mocked transcription and alignment so the package stays small
and testable. The optional WhisperX backend can be installed separately:

```bash
uv sync --extra ml
```

WhisperX also requires `ffmpeg` to be available on your system path.

```bash
wispr song.wav lyrics.txt --backend whisperx --model base --device auto --compute-type auto
```

Vocal separation is off by default. To run WhisperX on Demucs-isolated vocals, install
the optional separation extra and pass `--demucs`:

```bash
uv sync --extra ml --extra separation
wispr song.wav lyrics.txt --backend whisperx --demucs --model base --device auto
```

`--device auto` prefers CUDA when Torch reports that CUDA is available. With
`--compute-type auto`, wispr uses `float16` on CUDA and `int8` on CPU. If CUDA
is requested explicitly and unavailable, the run fails with a clear message.

Batch processing uses a CSV manifest with required `audio` and `lyrics` columns
and optional `output`, `language`, `title`, `artist`, and `album` columns.
Relative paths are resolved beside the manifest.

```csv
audio,lyrics,output,language
song-a.flac,song-a.txt,out/song-a.lrc,en
song-b.flac,song-b.txt,out/song-b.lrc,en
```

```bash
wispr batch manifest.csv --backend whisperx --device auto --compute-type auto --debug --force
```

Batch runs are sequential in one process so WhisperX models can be reused
without oversubscribing the GPU. Each run writes a `*.summary.json` file beside
the manifest with per-row status and stage timings.

Benchmark commands wrap the same pipeline and write a JSON report with runtime
configuration, alignment quality, warnings, stage timings, and total wall time.

```bash
wispr benchmark song.wav lyrics.txt --backend whisperx --device auto --compute-type auto --debug --force
wispr benchmark batch manifest.csv --backend whisperx --device auto --compute-type auto --debug --force
```

Single-song benchmark reports default to `<output-stem>.benchmark.json`. Batch
benchmark reports default to `<manifest-stem>.benchmark.json`.

The emitted `.lrc` still uses the supplied lyrics file as canonical text. WhisperX only
provides timing evidence.

For a local real-audio smoke run, place ignored files under `inputs/` and write outputs
back under that ignored tree:

For example:

```bash
uv run wispr inputs/03-giveon-twenties.flac inputs/lyrics.txt \
  --backend whisperx \
  --demucs \
  --model base \
  --device auto \
  --compute-type auto \
  --debug \
  --force \
  -o inputs/out/twenties.lrc
```

For local performance comparisons:

```bash
uv run wispr benchmark inputs/03-giveon-twenties.flac inputs/lyrics.txt --backend whisperx --device auto --compute-type auto --debug --force -o inputs/out/twenties.lrc
uv run wispr benchmark inputs/03-giveon-twenties.flac inputs/lyrics.txt --backend whisperx --demucs --device auto --compute-type auto --debug --force -o inputs/out/twenties-demucs.lrc
uv run wispr benchmark batch inputs/manifest.csv --backend whisperx --device auto --compute-type auto --debug --force
```

The WhisperX backend checks for both the optional Python dependency and `ffmpeg` before
running. Debug output includes raw backend payloads, normalized dataclass state, and an
alignment summary so failed or weak runs can be inspected without changing the `.lrc`
contract.

The code is organized around:

- typed dataclasses at stage boundaries
- deterministic LRC formatting
- a thin Typer CLI over reusable library code
- debug artifacts that mirror internal pipeline state
- structured warnings for weak alignment
- batch summaries and stage timings for runtime profiling
- benchmark reports for repeatable performance comparisons

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

Slow ML tests should be marked with `pytest.mark.ml` and run explicitly:

```bash
uv run pytest -m ml
```

The local ML smoke test is skipped during the default test suite and only runs when the
`inputs/` smoke files are present.
