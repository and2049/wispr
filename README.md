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
- `--device`: runtime device, defaulting to `cpu`
- `--compute-type`: WhisperX compute type, defaulting to `int8`
- `--language`: WhisperX language code, defaulting to `en`

## Current milestone

The default backend uses mocked transcription and alignment so the package stays small
and testable. The optional WhisperX backend can be installed separately:

```bash
uv sync --extra ml
```

WhisperX also requires `ffmpeg` to be available on your system path.

```bash
wispr song.wav lyrics.txt --backend whisperx --model base --device cpu --compute-type int8
```

Vocal separation is off by default. To run WhisperX on Demucs-isolated vocals, install
the optional separation extra and pass `--demucs`:

```bash
uv sync --extra ml --extra separation
wispr song.wav lyrics.txt --backend whisperx --demucs --model base --device cpu
```

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
  --device cpu \
  --compute-type int8 \
  --debug \
  --force \
  -o inputs/out/twenties.lrc
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
