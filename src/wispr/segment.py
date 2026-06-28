from __future__ import annotations

from wispr.models import AlignedWord, LyricLine, WisprWarning

WEAK_CONFIDENCE_THRESHOLD = 0.7


def segment_lines(
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
