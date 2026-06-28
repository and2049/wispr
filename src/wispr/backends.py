from __future__ import annotations

from pathlib import Path
from typing import Protocol

from wispr.models import AlignedWord, TrackMetadata, TranscriptWord


class MetadataReader(Protocol):
    def read(self, audio_path: Path) -> TrackMetadata: ...


class VocalSeparator(Protocol):
    def separate(self, audio_path: Path) -> Path: ...


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> tuple[TranscriptWord, ...]: ...


class Aligner(Protocol):
    def align(
        self,
        transcript: tuple[TranscriptWord, ...],
        lyrics: tuple[str, ...],
        audio_path: Path | None = None,
    ) -> tuple[AlignedWord, ...]: ...


class EmptyMetadataReader:
    def read(self, audio_path: Path) -> TrackMetadata:
        return TrackMetadata(source_path=audio_path)


class NoOpVocalSeparator:
    def separate(self, audio_path: Path) -> Path:
        return audio_path


class MockTranscriber:
    def transcribe(self, audio_path: Path) -> tuple[TranscriptWord, ...]:
        return ()


class MockAligner:
    def align(
        self,
        transcript: tuple[TranscriptWord, ...],
        lyrics: tuple[str, ...],
        audio_path: Path | None = None,
    ) -> tuple[AlignedWord, ...]:
        del transcript, audio_path
        words: list[AlignedWord] = []
        for line_index, line in enumerate(lyrics):
            start = line_index * 4.0
            tokens = line.split() or [""]
            for token_index, token in enumerate(tokens):
                confidence = 0.55 if line_index == 0 and token_index == 0 else 0.95
                words.append(
                    AlignedWord(
                        text=token,
                        start=start + token_index * 0.35,
                        end=start + token_index * 0.35 + 0.25,
                        confidence=confidence,
                        timestamp_source="mock",
                    )
                )
        return tuple(words)
