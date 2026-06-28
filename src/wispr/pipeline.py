from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from wispr.audio import ensure_output_writable, resolve_output_path, validate_audio_path
from wispr.backends import (
    Aligner,
    EmptyMetadataReader,
    MetadataReader,
    MockAligner,
    MockTranscriber,
    NoOpVocalSeparator,
    Transcriber,
    VocalSeparator,
)
from wispr.lrc import serialize_lrc
from wispr.lyrics import read_lyrics
from wispr.models import LrcDocument, WisprWarning, to_jsonable
from wispr.segment import segment_lines


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    warnings: tuple[WisprWarning, ...]
    debug_dir: Path | None = None


@dataclass(frozen=True)
class PipelineBackends:
    metadata: MetadataReader = EmptyMetadataReader()
    separator: VocalSeparator = NoOpVocalSeparator()
    transcriber: Transcriber = MockTranscriber()
    aligner: Aligner = MockAligner()


def run(
    audio_path: Path,
    lyrics_path: Path,
    *,
    output_path: Path | None = None,
    force: bool = False,
    debug: bool = False,
    separate_vocals: bool = True,
    backends: PipelineBackends | None = None,
) -> PipelineResult:
    backends = backends or PipelineBackends()
    audio_path = validate_audio_path(audio_path)
    output_path = resolve_output_path(audio_path, output_path)
    ensure_output_writable(output_path, force=force)

    lyrics = read_lyrics(lyrics_path)
    metadata = backends.metadata.read(audio_path)
    processing_audio = backends.separator.separate(audio_path) if separate_vocals else audio_path
    transcript = backends.transcriber.transcribe(processing_audio)
    alignment = backends.aligner.align(transcript, lyrics)
    lines, warnings = segment_lines(lyrics, alignment)

    document = LrcDocument(metadata=metadata, lines=lines)
    output_path.write_text(serialize_lrc(document), encoding="utf-8")
    debug_dir = write_debug(output_path, transcript, alignment, lines) if debug else None
    return PipelineResult(output_path=output_path, warnings=warnings, debug_dir=debug_dir)


def write_debug(output_path: Path, transcript: object, alignment: object, segments: object) -> Path:
    debug_dir = output_path.with_suffix("").with_name(f"{output_path.stem}.debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "transcript.json": transcript,
        "alignment.json": alignment,
        "segments.json": segments,
    }
    for name, value in artifacts.items():
        (debug_dir / name).write_text(
            json.dumps(to_jsonable(value), indent=2),
            encoding="utf-8",
        )
    return debug_dir
