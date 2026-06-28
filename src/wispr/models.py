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
    demucs_enabled: bool = False


@dataclass(frozen=True)
class BackendRuntimeConfig:
    backend: str = "mock"
    model_name: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    batch_size: int = 4
    language: str = "en"
    vad_method: str = "silero"
    demucs_enabled: bool = False
    separator_backend: str = "none"
    separator_model: str | None = None
    requested_device: str = "auto"
    requested_compute_type: str = "auto"
    runtime_note: str | None = None


@dataclass(frozen=True)
class TranscriptionConfig:
    model_name: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    batch_size: int = 4
    language: str = "en"
    vad_method: str = "silero"


@dataclass(frozen=True)
class AlignmentConfig:
    device: str = "auto"
    language: str = "en"
    return_char_alignments: bool = False


@dataclass(frozen=True)
class DemucsConfig:
    model_name: str = "htdemucs"
    device: str = "auto"


@dataclass(frozen=True)
class BatchJob:
    row_number: int
    audio_path: Path
    lyrics_path: Path
    output_path: Path | None = None
    language: str | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None


@dataclass(frozen=True)
class BatchJobResult:
    job: BatchJob
    status: str
    output_path: Path | None = None
    debug_dir: Path | None = None
    summary: AlignmentSummary | None = None
    warnings: tuple[WisprWarning, ...] = ()
    error_message: str | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchRunResult:
    manifest_path: Path
    summary_path: Path
    jobs: tuple[BatchJobResult, ...]
    total_jobs: int
    succeeded: int
    failed: int
    stage_timings: dict[str, float] = field(default_factory=dict)


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
    expected_tokens: int = 0
    matched_tokens: int = 0
    fuzzy_matches: int = 0
    interpolated: bool = False


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


@dataclass(frozen=True)
class AlignmentSummary:
    total_lyric_words: int
    aligned_words: int
    skipped_words: int
    average_confidence: float
    weak_line_count: int
    backend: str
    alignment_coverage: float = 0.0
    unmatched_lyric_tokens: int = 0
    unmatched_backend_tokens: int = 0
    fuzzy_matches: int = 0
    interpolated_lines: int = 0
    timestamp_sources: dict[str, int] = field(default_factory=dict)
    quality_warning: str | None = None


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
