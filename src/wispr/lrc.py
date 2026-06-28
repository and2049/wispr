from __future__ import annotations

from wispr.models import LrcDocument, TrackMetadata


def format_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    minutes, remainder = divmod(centiseconds, 6000)
    secs, cents = divmod(remainder, 100)
    return f"{minutes:02d}:{secs:02d}.{cents:02d}"


def metadata_tags(metadata: TrackMetadata) -> list[str]:
    tags = [
        ("ar", metadata.artist),
        ("al", metadata.album),
        ("ti", metadata.title),
    ]
    return [f"[{key}:{value}]" for key, value in tags if value]


def serialize_lrc(document: LrcDocument) -> str:
    lines = metadata_tags(document.metadata)
    lines.extend(
        f"[{format_timestamp(line.start)}]{line.text}"
        for line in document.lines
        if line.text != ""
    )
    return "\n".join(lines) + "\n"
