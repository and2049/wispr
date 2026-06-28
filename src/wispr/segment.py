from __future__ import annotations

from wispr.models import AlignedWord, AlignmentSummary, LyricLine, WisprWarning

WEAK_CONFIDENCE_THRESHOLD = 0.7


def segment_lines(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
) -> tuple[tuple[LyricLine, ...], tuple[WisprWarning, ...]]:
    return token_count_segment_lines(lyrics, words)


def token_count_segment_lines(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
) -> tuple[tuple[LyricLine, ...], tuple[WisprWarning, ...]]:
    cursor = 0
    lines: list[LyricLine] = []
    warnings: list[WisprWarning] = []

    for index, text in enumerate(lyrics, start=1):
        count = len(text.split()) or 1
        line_words = words[cursor : cursor + count]
        cursor += count
        line = make_line(index, text, line_words)
        lines.append(line)
        if line.confidence < WEAK_CONFIDENCE_THRESHOLD:
            warnings.append(
                WisprWarning(
                    line_number=index,
                    confidence=line.confidence,
                    timestamp_source=line.timestamp_source,
                    message=f"Line {index} has weak alignment confidence.",
                )
            )

    return tuple(lines), tuple(warnings)


def summarize_alignment(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
    warnings: tuple[WisprWarning, ...],
    *,
    backend: str,
    skipped_words: int = 0,
) -> AlignmentSummary:
    total_lyric_words = sum(len(line.split()) for line in lyrics)
    average_confidence = (
        round(sum(word.confidence for word in words) / len(words), 4) if words else 0.0
    )
    return AlignmentSummary(
        total_lyric_words=total_lyric_words,
        aligned_words=len(words),
        skipped_words=skipped_words,
        average_confidence=average_confidence,
        weak_line_count=len(warnings),
        backend=backend,
    )


def make_line(line_number: int, text: str, words: tuple[AlignedWord, ...]) -> LyricLine:
    if not words:
        return LyricLine(line_number=line_number, text=text)
    confidence = min(word.confidence for word in words)
    return LyricLine(
        line_number=line_number,
        text=text,
        words=words,
        start=words[0].start,
        confidence=confidence,
        timestamp_source=words[0].timestamp_source,
    )
