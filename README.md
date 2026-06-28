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
- `--no-separate-vocals`: skip the vocal-separation stage
- `--backend mock|whisperx`: choose mocked timing or the optional WhisperX backend
- `--model`: WhisperX model name, defaulting to `base`
- `--device`: runtime device, defaulting to `cpu`
- `--compute-type`: WhisperX compute type, defaulting to `int8`

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

The emitted `.lrc` still uses the supplied lyrics file as canonical text. WhisperX only
provides timing evidence.

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
