from __future__ import annotations

import re
from dataclasses import dataclass

from wispr.models import AlignedWord, AlignmentSummary, LyricLine, WisprWarning

WEAK_CONFIDENCE_THRESHOLD = 0.55
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
LOOKAHEAD = 8


@dataclass(frozen=True)
class WordPiece:
    token: str
    word_index: int
    word: AlignedWord


def segment_lines(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
) -> tuple[tuple[LyricLine, ...], tuple[WisprWarning, ...]]:
    return match_segment_lines(lyrics, words)


def match_segment_lines(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
) -> tuple[tuple[LyricLine, ...], tuple[WisprWarning, ...]]:
    pieces = aligned_word_pieces(words)
    piece_cursor = 0
    word_cursor = 0
    lines: list[LyricLine] = []
    warnings: list[WisprWarning] = []

    for index, text in enumerate(lyrics, start=1):
        expected = lyric_tokens(text)
        line_words, word_cursor, piece_cursor = match_line_words(
            expected,
            words,
            pieces,
            word_cursor,
            piece_cursor,
        )
        line = make_line(index, text, expected, line_words)
        lines.append(line)
        if expected and line.confidence < WEAK_CONFIDENCE_THRESHOLD:
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
    total_lyric_words = sum(len(lyric_tokens(line)) for line in lyrics)
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


def lyric_tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize_token(token) for token in WORD_RE.findall(text))


def normalize_token(token: str) -> str:
    return token.replace("'", "").lower()


def aligned_word_pieces(words: tuple[AlignedWord, ...]) -> tuple[WordPiece, ...]:
    pieces: list[WordPiece] = []
    for word_index, word in enumerate(words):
        for token in lyric_tokens(word.text):
            pieces.append(WordPiece(token=token, word_index=word_index, word=word))
    return tuple(pieces)


def match_line_words(
    expected: tuple[str, ...],
    words: tuple[AlignedWord, ...],
    pieces: tuple[WordPiece, ...],
    word_cursor: int,
    piece_cursor: int,
) -> tuple[tuple[AlignedWord, ...], int, int]:
    if not expected:
        return (), word_cursor, piece_cursor

    matched: list[AlignedWord] = []
    seen: set[int] = set()
    for token in expected:
        match_index = next_match_index(token, pieces, piece_cursor)
        if match_index is None:
            continue
        piece_cursor = match_index + 1
        piece = pieces[match_index]
        word_cursor = max(word_cursor, piece.word_index + 1)
        if piece.word_index not in seen:
            matched.append(piece.word)
            seen.add(piece.word_index)
    if matched:
        return tuple(matched), word_cursor, piece_cursor

    fallback_count = max(1, min(len(expected), len(words) - word_cursor))
    fallback = words[word_cursor : word_cursor + fallback_count]
    word_cursor += len(fallback)
    piece_cursor = piece_cursor_for_word(pieces, word_cursor)
    return tuple(fallback), word_cursor, piece_cursor


def next_match_index(token: str, pieces: tuple[WordPiece, ...], cursor: int) -> int | None:
    stop = min(len(pieces), cursor + LOOKAHEAD)
    for index in range(cursor, stop):
        if pieces[index].token == token:
            return index
    for index in range(stop, len(pieces)):
        if pieces[index].token == token:
            return index
    return None


def piece_cursor_for_word(pieces: tuple[WordPiece, ...], word_index: int) -> int:
    for index, piece in enumerate(pieces):
        if piece.word_index >= word_index:
            return index
    return len(pieces)


def make_line(
    line_number: int,
    text: str,
    expected: tuple[str, ...],
    words: tuple[AlignedWord, ...],
) -> LyricLine:
    if not expected:
        return LyricLine(line_number=line_number, text=text, confidence=1.0)
    if not words:
        return LyricLine(line_number=line_number, text=text)
    confidence = line_confidence(expected, words)
    return LyricLine(
        line_number=line_number,
        text=text,
        words=words,
        start=words[0].start,
        confidence=confidence,
        timestamp_source=words[0].timestamp_source,
    )


def line_confidence(expected: tuple[str, ...], words: tuple[AlignedWord, ...]) -> float:
    coverage = min(len(words) / len(expected), 1.0)
    average = robust_average_confidence(words)
    return round(coverage * average, 4)


def robust_average_confidence(words: tuple[AlignedWord, ...]) -> float:
    scores = sorted(word.confidence for word in words)
    if len(scores) >= 4:
        scores = scores[1:]
    return sum(scores) / len(scores)
