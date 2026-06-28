from __future__ import annotations

from importlib import import_module
from pathlib import Path
from shutil import which
from typing import Any

from wispr.models import (
    AlignedWord,
    AlignmentConfig,
    TranscriptionConfig,
    TranscriptWord,
)

INSTALL_MESSAGE = "WhisperX backend requires optional ML dependencies. Install with: wispr[ml]"
FFMPEG_MESSAGE = "WhisperX backend requires ffmpeg on PATH."


def import_whisperx() -> Any:
    try:
        return import_module("whisperx")
    except ImportError as error:
        raise RuntimeError(INSTALL_MESSAGE) from error


def validate_whisperx_runtime() -> None:
    import_whisperx()
    if which("ffmpeg") is None:
        raise RuntimeError(FFMPEG_MESSAGE)


class WhisperTranscriber:
    def __init__(self, config: TranscriptionConfig | None = None) -> None:
        self.config = config or TranscriptionConfig()
        self.last_raw_result: dict[str, Any] | None = None
        self.skipped_words = 0

    def transcribe(self, audio_path: Path) -> tuple[TranscriptWord, ...]:
        whisperx = import_whisperx()
        model = whisperx.load_model(
            self.config.model_name,
            self.config.device,
            compute_type=self.config.compute_type,
            language=self.config.language,
        )
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=self.config.batch_size)
        self.last_raw_result = result
        words, skipped = transcript_words_with_skips(result)
        self.skipped_words = skipped
        return words


class WhisperXAligner:
    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or AlignmentConfig()
        self.last_raw_result: dict[str, Any] | None = None
        self.skipped_words = 0

    def align(
        self,
        transcript: tuple[TranscriptWord, ...],
        lyrics: tuple[str, ...],
        audio_path: Path | None = None,
    ) -> tuple[AlignedWord, ...]:
        if audio_path is None:
            raise ValueError("WhisperX alignment requires the prepared audio path.")
        whisperx = import_whisperx()
        model, metadata = whisperx.load_align_model(
            language_code=self.config.language,
            device=self.config.device,
        )
        audio = whisperx.load_audio(str(audio_path))
        result = whisperx.align(
            canonical_segments(lyrics, transcript),
            model,
            metadata,
            audio,
            self.config.device,
            return_char_alignments=self.config.return_char_alignments,
        )
        self.last_raw_result = result
        words, skipped = aligned_words_with_skips(result)
        self.skipped_words = skipped
        return words


def transcript_words(result: dict[str, Any]) -> tuple[TranscriptWord, ...]:
    return transcript_words_with_skips(result)[0]


def transcript_words_with_skips(result: dict[str, Any]) -> tuple[tuple[TranscriptWord, ...], int]:
    words: list[TranscriptWord] = []
    skipped = 0
    for word in iter_word_dicts(result):
        text = word.get("word") or word.get("text")
        if not text or "start" not in word or "end" not in word:
            skipped += 1
            continue
        words.append(
            TranscriptWord(
                text=str(text).strip(),
                start=float(word["start"]),
                end=float(word["end"]),
                confidence=float(word.get("score", word.get("confidence", 1.0))),
                source="whisperx",
            )
        )
    return tuple(words), skipped


def aligned_words(result: dict[str, Any]) -> tuple[AlignedWord, ...]:
    return aligned_words_with_skips(result)[0]


def aligned_words_with_skips(result: dict[str, Any]) -> tuple[tuple[AlignedWord, ...], int]:
    words: list[AlignedWord] = []
    skipped = 0
    for word in iter_word_dicts(result):
        text = word.get("word") or word.get("text")
        if not text or "start" not in word or "end" not in word:
            skipped += 1
            continue
        words.append(
            AlignedWord(
                text=str(text).strip(),
                start=float(word["start"]),
                end=float(word["end"]),
                confidence=float(word.get("score", word.get("confidence", 1.0))),
                timestamp_source="whisperx",
            )
        )
    return tuple(words), skipped


def canonical_segments(
    lyrics: tuple[str, ...],
    transcript: tuple[TranscriptWord, ...],
) -> list[dict[str, Any]]:
    segment: dict[str, Any] = {"text": "\n".join(lyrics)}
    if transcript:
        segment["start"] = transcript[0].start
        segment["end"] = transcript[-1].end
    return [segment]


def iter_word_dicts(result: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        words.extend(segment.get("words", []))
    return words
