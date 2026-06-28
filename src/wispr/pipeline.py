from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TypeVar

from wispr.artifacts import (
    copy_artifact,
    demucs_artifact_path,
    demucs_config,
    file_fingerprint,
    load_manifest,
    matching_entry,
    read_transcript_artifact,
    record_entry,
    transcript_artifact_path,
    transcript_config,
    write_transcript_artifact,
)
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
from wispr.diagnostics import build_diagnostics
from wispr.lrc import serialize_lrc
from wispr.lyrics import read_lyrics, validate_lyrics_path
from wispr.matching import MatchingDiagnostics
from wispr.metadata import MutagenMetadataReader
from wispr.models import (
    AccuracyDiagnostics,
    AlignedWord,
    AlignmentSummary,
    ArtifactEntry,
    ArtifactManifest,
    BackendRuntimeConfig,
    LrcDocument,
    PipelineInputs,
    TrackMetadata,
    TranscriptWord,
    WisprWarning,
    to_jsonable,
)
from wispr.segment import segment_lines_with_diagnostics, summarize_alignment

T = TypeVar("T")


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    warnings: tuple[WisprWarning, ...]
    summary: AlignmentSummary
    debug_dir: Path | None = None
    processing_audio_path: Path | None = None
    stage_timings: dict[str, float] | None = None
    diagnostics: AccuracyDiagnostics = field(default_factory=AccuracyDiagnostics)


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
    reuse_artifacts: bool = False,
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
        reuse_artifacts=reuse_artifacts,
    )
    artifact_manifest = load_manifest(inputs.output_path)
    artifact_hits: list[str] = []
    artifact_misses: list[str] = []
    artifact_invalidations: list[str] = []
    lyrics = timed(timings, "lyrics", lambda: read_lyrics(inputs.lyrics_path))
    metadata = timed(
        timings,
        "metadata",
        lambda: read_metadata(backends.metadata, inputs.audio_path),
    )
    metadata = apply_metadata_override(metadata, metadata_override)
    processing_audio, artifact_manifest = timed(
        timings,
        "separation",
        lambda: prepare_audio(
            backends.separator,
            inputs,
            backends.runtime,
            artifact_manifest,
            artifact_hits,
            artifact_misses,
            artifact_invalidations,
        ),
    )
    transcript, artifact_manifest = timed(
        timings,
        "transcription",
        lambda: transcribe_audio(
            backends.transcriber,
            processing_audio,
            inputs,
            backends.runtime,
            artifact_manifest,
            artifact_hits,
            artifact_misses,
            artifact_invalidations,
        ),
    )
    ensure_transcript_quality(backends.runtime.backend, transcript)
    alignment = timed(
        timings,
        "alignment",
        lambda: align_lyrics(backends.aligner, transcript, lyrics, processing_audio),
    )
    ensure_alignment_quality(backends.runtime.backend, alignment)
    lines, warnings, summary, matching = timed(
        timings,
        "segmentation",
        lambda: segment_and_summarize(lyrics, alignment, backends),
    )

    timed(
        timings,
        "output",
        lambda: write_output(inputs.output_path, LrcDocument(metadata=metadata, lines=lines)),
    )
    diagnostics = build_diagnostics(
        summary=summary,
        lines=lines,
        warnings=warnings,
        fallback_words=(
            fallback_word_count(backends.transcriber) + fallback_word_count(backends.aligner)
        ),
        reuse_enabled=inputs.reuse_artifacts,
        artifact_hits=tuple(artifact_hits),
        artifact_misses=tuple(artifact_misses),
        artifact_invalidations=tuple(artifact_invalidations),
        artifact_paths=artifact_paths(inputs.output_path, backends.runtime),
        matching=matching,
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
            matching,
            diagnostics,
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
        diagnostics=diagnostics,
    )


def prepare_inputs(
    audio_path: Path,
    lyrics_path: Path,
    *,
    output_path: Path | None,
    force: bool,
    debug: bool,
    demucs_enabled: bool,
    reuse_artifacts: bool,
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
        reuse_artifacts=reuse_artifacts,
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


def prepare_audio(
    separator: VocalSeparator,
    inputs: PipelineInputs,
    runtime: BackendRuntimeConfig,
    manifest: ArtifactManifest,
    hits: list[str],
    misses: list[str],
    invalidations: list[str],
) -> tuple[Path, ArtifactManifest]:
    if not inputs.demucs_enabled:
        return inputs.audio_path, manifest
    fingerprint = file_fingerprint(inputs.audio_path)
    config = demucs_config(runtime)
    destination = demucs_artifact_path(inputs.output_path)
    if inputs.reuse_artifacts:
        entry, reason = matching_entry(
            manifest,
            "demucs",
            fingerprint=fingerprint,
            config=config,
        )
        if entry:
            hits.append("demucs")
            return entry.path, manifest
        misses.append("demucs")
        if reason:
            invalidations.append(reason)
    separated = separator.separate(inputs.audio_path, inputs.output_path)
    if not inputs.reuse_artifacts:
        return separated, manifest
    cached = copy_artifact(separated, destination)
    manifest = record_entry(
        inputs.output_path,
        manifest,
        ArtifactEntry(kind="demucs", path=cached, fingerprint=fingerprint, config=config),
    )
    return cached, manifest


def transcribe_audio(
    transcriber: Transcriber,
    audio_path: Path,
    inputs: PipelineInputs,
    runtime: BackendRuntimeConfig,
    manifest: ArtifactManifest,
    hits: list[str],
    misses: list[str],
    invalidations: list[str],
) -> tuple[tuple[TranscriptWord, ...], ArtifactManifest]:
    can_reuse = inputs.reuse_artifacts and runtime.backend == "whisperx"
    if can_reuse:
        fingerprint = file_fingerprint(audio_path)
        config = transcript_config(runtime)
        destination = transcript_artifact_path(inputs.output_path)
        entry, reason = matching_entry(
            manifest,
            "transcript",
            fingerprint=fingerprint,
            config=config,
        )
        if entry:
            try:
                transcript, skipped, fallback, raw = read_transcript_artifact(entry.path)
            except (KeyError, OSError, TypeError, ValueError) as error:
                misses.append("transcript")
                invalidations.append(f"transcript: unreadable artifact ({error})")
            else:
                hits.append("transcript")
                set_backend_debug(transcriber, raw, skipped, fallback)
                return transcript, manifest
        else:
            misses.append("transcript")
            if reason:
                invalidations.append(reason)
    transcript = transcriber.transcribe(audio_path)
    if not can_reuse:
        return transcript, manifest
    fingerprint = file_fingerprint(audio_path)
    config = transcript_config(runtime)
    destination = transcript_artifact_path(inputs.output_path)
    write_transcript_artifact(
        destination,
        transcript,
        raw=getattr(transcriber, "last_raw_result", None),
        skipped_words=skipped_word_count(transcriber),
        fallback_words=fallback_word_count(transcriber),
    )
    manifest = record_entry(
        inputs.output_path,
        manifest,
        ArtifactEntry(kind="transcript", path=destination, fingerprint=fingerprint, config=config),
    )
    return transcript, manifest


def set_backend_debug(backend: object, raw: object, skipped: int, fallback: int) -> None:
    backend.last_raw_result = raw
    backend.skipped_words = skipped
    backend.fallback_words = fallback


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
) -> tuple[object, tuple[WisprWarning, ...], AlignmentSummary, MatchingDiagnostics]:
    lines, warnings, matching = segment_lines_with_diagnostics(lyrics, alignment)
    summary = summarize_alignment(
        lyrics,
        alignment,
        lines,
        warnings,
        backend=backends.runtime.backend,
        skipped_words=skipped_word_count(backends.transcriber)
        + skipped_word_count(backends.aligner),
    )
    return lines, warnings, summary, matching


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
    matching: MatchingDiagnostics,
    diagnostics: AccuracyDiagnostics,
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
        "segments.json": {"segments": segments, "summary": summary, "matching": matching},
        "diagnostics.json": diagnostics,
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


def artifact_paths(output_path: Path, runtime: BackendRuntimeConfig) -> dict[str, Path]:
    artifact_dir = output_path.with_suffix("").with_name(f"{output_path.stem}.artifacts")
    paths = {"manifest": artifact_dir / "manifest.json"}
    if runtime.demucs_enabled:
        paths["demucs"] = demucs_artifact_path(output_path)
    if runtime.backend == "whisperx":
        paths["transcript"] = transcript_artifact_path(output_path)
    return paths


def ensure_transcript_quality(backend: str, transcript: tuple[TranscriptWord, ...]) -> None:
    if backend != "mock" and not transcript:
        raise ValueError("Transcription produced no timed words.")


def ensure_alignment_quality(backend: str, alignment: tuple[AlignedWord, ...]) -> None:
    if backend != "mock" and not alignment:
        raise ValueError("Alignment produced no timed words.")
