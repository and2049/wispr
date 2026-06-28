from __future__ import annotations

from pathlib import Path
from typing import Any

from mutagen import File

from wispr.backends import EmptyMetadataReader, MetadataReader
from wispr.models import TrackMetadata


class MutagenMetadataReader:
    def read(self, audio_path: Path) -> TrackMetadata:
        tags = read_tags(audio_path)
        return TrackMetadata(
            title=first_tag(tags, "title", "TIT2", "\u00a9nam"),
            artist=first_tag(tags, "artist", "TPE1", "\u00a9ART"),
            album=first_tag(tags, "album", "TALB", "\u00a9alb"),
            source_path=audio_path,
        )


def read_tags(audio_path: Path) -> Any:
    try:
        audio = File(audio_path, easy=True)
    except Exception:
        return None
    return getattr(audio, "tags", None) if audio else None


def first_tag(tags: Any, *keys: str) -> str | None:
    if not tags:
        return None
    for key in keys:
        value = tags.get(key)
        if isinstance(value, list | tuple):
            value = value[0] if value else None
        if value:
            return str(value)
    return None


__all__ = ["EmptyMetadataReader", "MetadataReader", "MutagenMetadataReader"]
