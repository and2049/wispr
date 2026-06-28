from __future__ import annotations

from pathlib import Path


def validate_lyrics_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Lyrics file does not exist: {path}.")
    if not path.is_file():
        raise ValueError(f"Lyrics path is not a file: {path}.")
    if path.read_text(encoding="utf-8").strip() == "":
        raise ValueError(f"Lyrics file is empty: {path}.")
    return path


def read_lyrics(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    return tuple(text.splitlines())
