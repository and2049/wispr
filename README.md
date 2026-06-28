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

## Current milestone

This foundation uses mocked metadata, transcription, alignment, and vocal-separation
adapters. That keeps the library small and testable while preserving the integration
points needed for future WhisperX or vocal-separation backends.

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
