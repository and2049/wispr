from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrackMetadata:
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class PipelineInputs:
    audio_path: Path
    lyrics_path: Path
    output_path: Path
    force: bool = False
    debug: bool = False
    separate_vocals: bool = True


@dataclass(frozen=True)
class TranscriptionConfig:
    model_name: str = "base"


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start: float
    end: float
    confidence: float = 1.0
    source: str = "mock"


@dataclass(frozen=True)
class AlignedWord:
    text: str
    start: float
    end: float
    confidence: float = 1.0
    timestamp_source: str = "mock"


@dataclass(frozen=True)
class LyricLine:
    line_number: int
    text: str
    words: tuple[AlignedWord, ...] = ()
    start: float = 0.0
    confidence: float = 0.0
    timestamp_source: str = "unaligned"


@dataclass(frozen=True)
class LrcDocument:
    metadata: TrackMetadata = field(default_factory=TrackMetadata)
    lines: tuple[LyricLine, ...] = ()


@dataclass(frozen=True)
class WisprWarning:
    line_number: int
    confidence: float
    timestamp_source: str
    message: str


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value
