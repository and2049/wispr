from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
    LrcDocument,
    PipelineInputs,
    TrackMetadata,
    TranscriptWord,
    WisprWarning,
    to_jsonable,
)
from wispr.segment import segment_lines


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    warnings: tuple[WisprWarning, ...]
    debug_dir: Path | None = None


@dataclass(frozen=True)
class PipelineBackends:
    metadata: MetadataReader = MutagenMetadataReader()
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
    inputs = prepare_inputs(
        audio_path,
        lyrics_path,
        output_path=output_path,
        force=force,
        debug=debug,
        separate_vocals=separate_vocals,
    )
    lyrics = read_lyrics(inputs.lyrics_path)
    metadata = read_metadata(backends.metadata, inputs.audio_path)
    processing_audio = prepare_audio(backends.separator, inputs)
    transcript = transcribe_audio(backends.transcriber, processing_audio)
    alignment = align_lyrics(backends.aligner, transcript, lyrics)
    lines, warnings = segment_lines(lyrics, alignment)

    write_output(inputs.output_path, LrcDocument(metadata=metadata, lines=lines))
    debug_dir = (
        write_debug(inputs, metadata, transcript, alignment, lines)
        if inputs.debug
        else None
    )
    return PipelineResult(output_path=inputs.output_path, warnings=warnings, debug_dir=debug_dir)


def prepare_inputs(
    audio_path: Path,
    lyrics_path: Path,
    *,
    output_path: Path | None,
    force: bool,
    debug: bool,
    separate_vocals: bool,
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
        separate_vocals=separate_vocals,
    )


def read_metadata(reader: MetadataReader, audio_path: Path) -> TrackMetadata:
    return reader.read(audio_path)


def prepare_audio(separator: VocalSeparator, inputs: PipelineInputs) -> Path:
    return separator.separate(inputs.audio_path) if inputs.separate_vocals else inputs.audio_path


def transcribe_audio(transcriber: Transcriber, audio_path: Path) -> tuple[TranscriptWord, ...]:
    return transcriber.transcribe(audio_path)


def align_lyrics(
    aligner: Aligner,
    transcript: tuple[TranscriptWord, ...],
    lyrics: tuple[str, ...],
) -> tuple[AlignedWord, ...]:
    return aligner.align(transcript, lyrics)


def write_output(output_path: Path, document: LrcDocument) -> None:
    output_path.write_text(serialize_lrc(document), encoding="utf-8")


def write_debug(
    inputs: PipelineInputs,
    metadata: TrackMetadata,
    transcript: object,
    alignment: object,
    segments: object,
) -> Path:
    debug_dir = inputs.output_path.with_suffix("").with_name(f"{inputs.output_path.stem}.debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "inputs.json": {"inputs": inputs, "metadata": metadata},
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
