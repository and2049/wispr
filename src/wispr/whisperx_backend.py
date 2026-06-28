from __future__ import annotations

import io
import math
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
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
TORCHCODEC_WARNING = (
    r"\s*torchcodec is not installed correctly so built-in audio decoding will fail\."
)
PART_RE = re.compile(r"\S+")


def import_whisperx() -> Any:
    try:
        return import_module("whisperx")
    except ImportError as error:
        raise RuntimeError(INSTALL_MESSAGE) from error


def suppress_runtime_warnings() -> None:
    warnings.filterwarnings("ignore", message=TORCHCODEC_WARNING, category=UserWarning)


def quiet_stdout() -> io.StringIO:
    return io.StringIO()


def quiet_streams() -> tuple[io.StringIO, io.StringIO]:
    return io.StringIO(), io.StringIO()


def validate_whisperx_runtime() -> None:
    suppress_runtime_warnings()
    import_whisperx()
    if which("ffmpeg") is None:
        raise RuntimeError(FFMPEG_MESSAGE)


class WhisperTranscriber:
    def __init__(self, config: TranscriptionConfig | None = None) -> None:
        self.config = config or TranscriptionConfig()
        self.last_raw_result: dict[str, Any] | None = None
        self.skipped_words = 0
        self.fallback_words = 0
        self._model: Any | None = None

    def transcribe(self, audio_path: Path) -> tuple[TranscriptWord, ...]:
        whisperx = import_whisperx()
        audio = whisperx.load_audio(str(audio_path))
        stdout, stderr = quiet_streams()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = self.model().transcribe(audio, batch_size=self.config.batch_size)
        self.last_raw_result = result
        words, skipped, fallback = transcript_words_with_skips(result)
        self.skipped_words = skipped
        self.fallback_words = fallback
        return words

    def model(self) -> Any:
        if self._model is None:
            suppress_runtime_warnings()
            whisperx = import_whisperx()
            stdout, stderr = quiet_streams()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self._model = whisperx.load_model(
                    self.config.model_name,
                    self.config.device,
                    compute_type=self.config.compute_type,
                    language=self.config.language,
                    vad_method=self.config.vad_method,
                )
        return self._model


class WhisperXAligner:
    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or AlignmentConfig()
        self.last_raw_result: dict[str, Any] | None = None
        self.skipped_words = 0
        self.fallback_words = 0
        self._model_cache: dict[tuple[str, str], tuple[Any, Any]] = {}

    def align(
        self,
        transcript: tuple[TranscriptWord, ...],
        lyrics: tuple[str, ...],
        audio_path: Path | None = None,
    ) -> tuple[AlignedWord, ...]:
        if audio_path is None:
            raise ValueError("WhisperX alignment requires the prepared audio path.")
        whisperx = import_whisperx()
        model, metadata = self.alignment_model(self.config.language)
        audio = whisperx.load_audio(str(audio_path))
        stdout, stderr = quiet_streams()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = whisperx.align(
                canonical_segments(lyrics, transcript),
                model,
                metadata,
                audio,
                self.config.device,
                return_char_alignments=self.config.return_char_alignments,
            )
        self.last_raw_result = result
        words, skipped, fallback = aligned_words_with_skips(result)
        self.skipped_words = skipped
        self.fallback_words = fallback
        return words or raw_transcript_alignment(transcript)

    def alignment_model(self, language: str) -> tuple[Any, Any]:
        key = (language, self.config.device)
        if key not in self._model_cache:
            suppress_runtime_warnings()
            whisperx = import_whisperx()
            stdout, stderr = quiet_streams()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self._model_cache[key] = whisperx.load_align_model(
                    language_code=language,
                    device=self.config.device,
                )
        return self._model_cache[key]


def transcript_words(result: dict[str, Any]) -> tuple[TranscriptWord, ...]:
    return transcript_words_with_skips(result)[0]


def transcript_words_with_skips(
    result: dict[str, Any],
) -> tuple[tuple[TranscriptWord, ...], int, int]:
    words: list[TranscriptWord] = []
    skipped = 0
    fallback = 0
    for segment, word in iter_segment_words(result):
        text = word.get("word") or word.get("text")
        start, end = word_times(word, segment)
        if not text or start is None or end is None:
            skipped += 1
            continue
        if "start" not in word or "end" not in word:
            fallback += 1
        words.extend(
            split_transcript_word(
                str(text).strip(),
                start=start,
                end=end,
                confidence=word_confidence(word),
                source="whisperx",
            )
        )
    if words:
        return tuple(words), skipped, fallback

    for segment in result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text or "start" not in segment or "end" not in segment:
            continue
        words.append(
            TranscriptWord(
                text=text,
                start=float(segment["start"]),
                end=float(segment["end"]),
                confidence=logprob_confidence(segment.get("avg_logprob")),
                source="whisperx-segment",
            )
        )
    return tuple(words), skipped, fallback


def aligned_words(result: dict[str, Any]) -> tuple[AlignedWord, ...]:
    return aligned_words_with_skips(result)[0]


def aligned_words_with_skips(
    result: dict[str, Any],
) -> tuple[tuple[AlignedWord, ...], int, int]:
    words: list[AlignedWord] = []
    skipped = 0
    fallback = 0
    for segment, word in iter_segment_words(result):
        text = word.get("word") or word.get("text")
        start, end = word_times(word, segment)
        if not text or start is None or end is None:
            skipped += 1
            continue
        if "start" not in word or "end" not in word:
            fallback += 1
        words.extend(
            split_aligned_word(
                str(text).strip(),
                start=start,
                end=end,
                confidence=word_confidence(word),
                timestamp_source=(
                    "whisperx" if "start" in word and "end" in word else "whisperx-segment"
                ),
            )
        )
    return tuple(words), skipped, fallback


def canonical_segments(
    lyrics: tuple[str, ...],
    transcript: tuple[TranscriptWord, ...],
) -> list[dict[str, Any]]:
    if uses_segment_fallback(transcript):
        return transcript_segment_blocks(lyrics, transcript)

    segment: dict[str, Any] = {"text": "\n".join(lyrics)}
    if transcript:
        segment["start"] = transcript[0].start
        segment["end"] = transcript[-1].end
    return [segment]


def iter_segment_words(result: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    words: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for segment in result.get("segments", []):
        words.extend((segment, word) for word in segment.get("words", []))
    return words


def word_times(word: dict[str, Any], segment: dict[str, Any]) -> tuple[float | None, float | None]:
    start = word.get("start", segment.get("start"))
    end = word.get("end", segment.get("end"))
    if start is None or end is None:
        return None, None
    return float(start), float(end)


def word_confidence(word: dict[str, Any], default: float = 0.5) -> float:
    value = word.get("score", word.get("confidence"))
    if value is None:
        return default
    return max(0.0, min(float(value), 1.0))


def logprob_confidence(value: Any, default: float = 0.5) -> float:
    if value is None:
        return default
    return max(0.0, min(math.exp(float(value)), 1.0))


def raw_transcript_alignment(transcript: tuple[TranscriptWord, ...]) -> tuple[AlignedWord, ...]:
    return tuple(
        AlignedWord(
            text=word.text,
            start=word.start,
            end=word.end,
            confidence=word.confidence,
            timestamp_source="raw-whisper",
        )
        for word in transcript
    )


def split_transcript_word(
    text: str,
    *,
    start: float,
    end: float,
    confidence: float,
    source: str,
) -> tuple[TranscriptWord, ...]:
    return tuple(
        TranscriptWord(
            text=part,
            start=part_start,
            end=part_end,
            confidence=confidence,
            source=source,
        )
        for part, part_start, part_end in timed_parts(text, start, end)
    )


def split_aligned_word(
    text: str,
    *,
    start: float,
    end: float,
    confidence: float,
    timestamp_source: str,
) -> tuple[AlignedWord, ...]:
    return tuple(
        AlignedWord(
            text=part,
            start=part_start,
            end=part_end,
            confidence=confidence,
            timestamp_source=timestamp_source,
        )
        for part, part_start, part_end in timed_parts(text, start, end)
    )


def timed_parts(text: str, start: float, end: float) -> tuple[tuple[str, float, float], ...]:
    parts = PART_RE.findall(text)
    if len(parts) <= 1:
        return ((text, start, end),)
    duration = max(end - start, 0.0)
    step = duration / len(parts)
    timed: list[tuple[str, float, float]] = []
    for index, part in enumerate(parts):
        part_start = start + step * index
        part_end = end if index == len(parts) - 1 else start + step * (index + 1)
        timed.append((part, part_start, part_end))
    return tuple(timed)


def uses_segment_fallback(transcript: tuple[TranscriptWord, ...]) -> bool:
    return bool(transcript) and all(word.source == "whisperx-segment" for word in transcript)


def transcript_segment_blocks(
    lyrics: tuple[str, ...],
    transcript: tuple[TranscriptWord, ...],
) -> list[dict[str, Any]]:
    groups = lyric_line_groups(lyrics, transcript)
    segments: list[dict[str, Any]] = []
    for entry, group in zip(transcript, groups, strict=True):
        segments.append({"text": "\n".join(group), "start": entry.start, "end": entry.end})
    return segments


def lyric_line_groups(
    lyrics: tuple[str, ...],
    transcript: tuple[TranscriptWord, ...],
) -> list[tuple[str, ...]]:
    if not transcript:
        return [lyrics]

    groups: list[tuple[str, ...]] = []
    line_index = 0
    for segment_index, entry in enumerate(transcript):
        remaining_segments = len(transcript) - segment_index
        remaining_lines = len(lyrics) - line_index
        if segment_index == len(transcript) - 1 or remaining_lines <= remaining_segments:
            groups.append(lyrics[line_index:])
            break

        target = max(1, len(entry.text.split()))
        start = line_index
        seen = 0
        while line_index < len(lyrics) - (remaining_segments - 1):
            seen += max(1, len(lyrics[line_index].split()))
            line_index += 1
            if seen >= target:
                break
        groups.append(lyrics[start:line_index])
    return groups
