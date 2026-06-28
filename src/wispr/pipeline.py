from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeVar

from wispr.audio import ensure_output_writable, resolve_output_path, validate_audio_path
from wispr.backends import (
    Aligner,
    MetadataReader,
    MockAligner,
    MockTranscriber,
    NoOpVocalSeparator,
    Transcriber,
    VocalSeparator,
)
from wispr.lrc import serialize_lrc
from wispr.lyrics import read_lyrics, validate_lyrics_path
from wispr.metadata import MutagenMetadataReader
from wispr.models import (
    AlignedWord,
    AlignmentSummary,
    BackendRuntimeConfig,
    LrcDocument,
    PipelineInputs,
    TrackMetadata,
    TranscriptWord,
    WisprWarning,
    to_jsonable,
)
from wispr.segment import segment_lines, summarize_alignment

T = TypeVar("T")


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    warnings: tuple[WisprWarning, ...]
    summary: AlignmentSummary
    debug_dir: Path | None = None
    processing_audio_path: Path | None = None
    stage_timings: dict[str, float] | None = None


@dataclass(frozen=True)
class PipelineBackends:
    metadata: MetadataReader = MutagenMetadataReader()
    separator: VocalSeparator = NoOpVocalSeparator()
    transcriber: Transcriber = MockTranscriber()
    aligner: Aligner = MockAligner()
    runtime: BackendRuntimeConfig = BackendRuntimeConfig()


def run(
    audio_path: Path,
    lyrics_path: Path,
    *,
    output_path: Path | None = None,
    force: bool = False,
    debug: bool = False,
    demucs_enabled: bool = False,
    metadata_override: TrackMetadata | None = None,
    backends: PipelineBackends | None = None,
) -> PipelineResult:
    backends = backends or PipelineBackends()
    timings: dict[str, float] = {}
    inputs = prepare_inputs(
        audio_path,
        lyrics_path,
        output_path=output_path,
        force=force,
        debug=debug,
        demucs_enabled=demucs_enabled,
    )
    lyrics = timed(timings, "lyrics", lambda: read_lyrics(inputs.lyrics_path))
    metadata = timed(
        timings,
        "metadata",
        lambda: read_metadata(backends.metadata, inputs.audio_path),
    )
    metadata = apply_metadata_override(metadata, metadata_override)
    processing_audio = timed(
        timings,
        "separation",
        lambda: prepare_audio(backends.separator, inputs),
    )
    transcript = timed(
        timings,
        "transcription",
        lambda: transcribe_audio(backends.transcriber, processing_audio),
    )
    ensure_transcript_quality(backends.runtime.backend, transcript)
    alignment = timed(
        timings,
        "alignment",
        lambda: align_lyrics(backends.aligner, transcript, lyrics, processing_audio),
    )
    ensure_alignment_quality(backends.runtime.backend, alignment)
    lines, warnings, summary = timed(
        timings,
        "segmentation",
        lambda: segment_and_summarize(lyrics, alignment, backends),
    )

    timed(
        timings,
        "output",
        lambda: write_output(inputs.output_path, LrcDocument(metadata=metadata, lines=lines)),
    )
    debug_dir = (
        write_debug(
            inputs,
            backends.runtime,
            metadata,
            processing_audio,
            transcript,
            alignment,
            lines,
            summary,
            backends,
            timings,
        )
        if inputs.debug
        else None
    )
    return PipelineResult(
        output_path=inputs.output_path,
        warnings=warnings,
        summary=summary,
        debug_dir=debug_dir,
        processing_audio_path=processing_audio,
        stage_timings=timings,
    )


def prepare_inputs(
    audio_path: Path,
    lyrics_path: Path,
    *,
    output_path: Path | None,
    force: bool,
    debug: bool,
    demucs_enabled: bool,
) -> PipelineInputs:
    audio_path = validate_audio_path(audio_path)
    lyrics_path = validate_lyrics_path(lyrics_path)
    output_path = resolve_output_path(audio_path, output_path)
    ensure_output_writable(output_path, force=force)
    return PipelineInputs(
        audio_path=audio_path,
        lyrics_path=lyrics_path,
        output_path=output_path,
        force=force,
        debug=debug,
        demucs_enabled=demucs_enabled,
    )


def read_metadata(reader: MetadataReader, audio_path: Path) -> TrackMetadata:
    return reader.read(audio_path)


def apply_metadata_override(
    metadata: TrackMetadata,
    override: TrackMetadata | None,
) -> TrackMetadata:
    if override is None:
        return metadata
    return TrackMetadata(
        title=override.title or metadata.title,
        artist=override.artist or metadata.artist,
        album=override.album or metadata.album,
        source_path=metadata.source_path,
    )


def prepare_audio(separator: VocalSeparator, inputs: PipelineInputs) -> Path:
    if not inputs.demucs_enabled:
        return inputs.audio_path
    return separator.separate(inputs.audio_path, inputs.output_path)


def transcribe_audio(transcriber: Transcriber, audio_path: Path) -> tuple[TranscriptWord, ...]:
    return transcriber.transcribe(audio_path)


def align_lyrics(
    aligner: Aligner,
    transcript: tuple[TranscriptWord, ...],
    lyrics: tuple[str, ...],
    audio_path: Path,
) -> tuple[AlignedWord, ...]:
    return aligner.align(transcript, lyrics, audio_path)


def segment_and_summarize(
    lyrics: tuple[str, ...],
    alignment: tuple[AlignedWord, ...],
    backends: PipelineBackends,
) -> tuple[object, tuple[WisprWarning, ...], AlignmentSummary]:
    lines, warnings = segment_lines(lyrics, alignment)
    summary = summarize_alignment(
        lyrics,
        alignment,
        lines,
        warnings,
        backend=backends.runtime.backend,
        skipped_words=skipped_word_count(backends.transcriber)
        + skipped_word_count(backends.aligner),
    )
    return lines, warnings, summary


def write_output(output_path: Path, document: LrcDocument) -> None:
    output_path.write_text(serialize_lrc(document), encoding="utf-8")


def write_debug(
    inputs: PipelineInputs,
    runtime: BackendRuntimeConfig,
    metadata: TrackMetadata,
    processing_audio: Path,
    transcript: tuple[TranscriptWord, ...],
    alignment: tuple[AlignedWord, ...],
    segments: object,
    summary: AlignmentSummary,
    backends: PipelineBackends,
    timings: dict[str, float],
) -> Path:
    start = perf_counter()
    debug_dir = inputs.output_path.with_suffix("").with_name(f"{inputs.output_path.stem}.debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "inputs.json": {
            "inputs": inputs,
            "runtime": runtime,
            "metadata": metadata,
            "source_audio_path": inputs.audio_path,
            "processing_audio_path": processing_audio,
            "separator": debug_payload(processing_audio, backends.separator),
        },
        "transcript.json": debug_payload(transcript, backends.transcriber),
        "alignment.json": debug_payload(alignment, backends.aligner),
        "segments.json": {"segments": segments, "summary": summary},
    }
    for name, value in artifacts.items():
        (debug_dir / name).write_text(
            json.dumps(to_jsonable(value), indent=2),
            encoding="utf-8",
        )
    timings["debug"] = perf_counter() - start
    (debug_dir / "timings.json").write_text(
        json.dumps(to_jsonable(timings), indent=2),
        encoding="utf-8",
    )
    return debug_dir


def timed(timings: dict[str, float], stage: str, call: Callable[[], T]) -> T:
    start = perf_counter()
    try:
        return call()
    finally:
        timings[stage] = perf_counter() - start


def debug_payload(normalized: object, backend: object) -> dict[str, object]:
    return {
        "raw": getattr(backend, "last_raw_result", None),
        "normalized": normalized,
        "skipped_words": skipped_word_count(backend),
        "fallback_words": fallback_word_count(backend),
    }


def skipped_word_count(backend: object) -> int:
    return int(getattr(backend, "skipped_words", 0))


def fallback_word_count(backend: object) -> int:
    return int(getattr(backend, "fallback_words", 0))


def ensure_transcript_quality(backend: str, transcript: tuple[TranscriptWord, ...]) -> None:
    if backend != "mock" and not transcript:
        raise ValueError("Transcription produced no timed words.")


def ensure_alignment_quality(backend: str, alignment: tuple[AlignedWord, ...]) -> None:
    if backend != "mock" and not alignment:
        raise ValueError("Alignment produced no timed words.")
