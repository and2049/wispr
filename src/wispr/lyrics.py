from __future__ import annotations

from pathlib import Path


def read_lyrics(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    return tuple(text.splitlines())
