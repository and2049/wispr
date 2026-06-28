from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from wispr.models import AlignedWord, AlignmentSummary, LyricLine, WisprWarning

WEAK_CONFIDENCE_THRESHOLD = 0.55
FUZZY_MATCH_THRESHOLD = 0.84
INSERT_DELETE_COST = 1.0
MISMATCH_COST = 2.0


@dataclass(frozen=True)
class WordPiece:
    token: str
    word_index: int
    word: AlignedWord


@dataclass(frozen=True)
class CanonicalToken:
    token: str
    line_number: int


def segment_lines(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
) -> tuple[tuple[LyricLine, ...], tuple[WisprWarning, ...]]:
    canonical = canonical_tokens(lyrics)
    pieces = aligned_word_pieces(words)
    matches, fuzzy = align_token_indices(canonical, pieces)
    lines = interpolate_unmatched_lines(
        build_lines(lyrics, words, canonical, pieces, matches, fuzzy),
        words,
    )
    warnings = tuple(
        WisprWarning(
            line_number=line.line_number,
            confidence=line.confidence,
            timestamp_source=line.timestamp_source,
            message=f"Line {line.line_number} has weak alignment confidence.",
        )
        for line in lines
        if line.expected_tokens and line.confidence < WEAK_CONFIDENCE_THRESHOLD
    )
    return lines, warnings


def summarize_alignment(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
    lines: tuple[LyricLine, ...],
    warnings: tuple[WisprWarning, ...],
    *,
    backend: str,
    skipped_words: int = 0,
) -> AlignmentSummary:
    total_lyric_words = sum(line.expected_tokens for line in lines)
    matched_tokens = sum(line.matched_tokens for line in lines)
    token_count = len(aligned_word_pieces(words))
    average_confidence = (
        round(sum(word.confidence for word in words) / len(words), 4) if words else 0.0
    )
    coverage = round(matched_tokens / total_lyric_words, 4) if total_lyric_words else 0.0
    quality_warning = None
    if total_lyric_words and coverage < 0.5:
        quality_warning = (
            f"Low lyric coverage: matched {matched_tokens}/{total_lyric_words} lyric tokens."
        )
    return AlignmentSummary(
        total_lyric_words=total_lyric_words,
        aligned_words=len(words),
        skipped_words=skipped_words,
        average_confidence=average_confidence,
        weak_line_count=len(warnings),
        backend=backend,
        alignment_coverage=coverage,
        unmatched_lyric_tokens=max(total_lyric_words - matched_tokens, 0),
        unmatched_backend_tokens=max(token_count - matched_tokens, 0),
        fuzzy_matches=sum(line.fuzzy_matches for line in lines),
        interpolated_lines=sum(line.interpolated for line in lines if line.expected_tokens),
        timestamp_sources=timestamp_source_counts(lines),
        quality_warning=quality_warning,
    )


def lyric_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in (normalize_token(part) for part in split_tokens(text)) if token)


def split_tokens(text: str) -> tuple[str, ...]:
    chars: list[str] = []
    tokens: list[str] = []
    for char in normalize_apostrophes(text):
        if is_token_char(char):
            chars.append(char)
            continue
        if chars:
            tokens.append("".join(chars))
            chars.clear()
    if chars:
        tokens.append("".join(chars))
    return tuple(tokens)


def normalize_apostrophes(text: str) -> str:
    return text.translate(str.maketrans({"’": "'", "`": "'", "´": "'", "ʼ": "'"}))


def is_token_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "N"} or char == "'"


def normalize_token(token: str) -> str:
    folded = unicodedata.normalize("NFKD", token.casefold().replace("'", ""))
    return "".join(char for char in folded if unicodedata.category(char) != "Mn")


@lru_cache(maxsize=2048)
def token_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SequenceMatcher(a=left, b=right).ratio()


def canonical_tokens(lyrics: tuple[str, ...]) -> tuple[CanonicalToken, ...]:
    return tuple(
        CanonicalToken(token=token, line_number=line_number)
        for line_number, text in enumerate(lyrics, start=1)
        for token in lyric_tokens(text)
    )


def aligned_word_pieces(words: tuple[AlignedWord, ...]) -> tuple[WordPiece, ...]:
    pieces: list[WordPiece] = []
    for word_index, word in enumerate(words):
        for token in lyric_tokens(word.text):
            pieces.append(WordPiece(token=token, word_index=word_index, word=word))
    return tuple(pieces)


def align_token_indices(
    canonical: tuple[CanonicalToken, ...],
    pieces: tuple[WordPiece, ...],
) -> tuple[dict[int, int], dict[int, float]]:
    rows = len(canonical) + 1
    cols = len(pieces) + 1
    costs = [[0.0] * cols for _ in range(rows)]
    steps = [[""] * cols for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row * INSERT_DELETE_COST
        steps[row][0] = "up"
    for col in range(1, cols):
        costs[0][col] = col * INSERT_DELETE_COST
        steps[0][col] = "left"
    for row in range(1, rows):
        for col in range(1, cols):
            similarity = token_similarity(canonical[row - 1].token, pieces[col - 1].token)
            diagonal = costs[row - 1][col - 1] + substitution_cost(similarity)
            up = costs[row - 1][col] + INSERT_DELETE_COST
            left = costs[row][col - 1] + INSERT_DELETE_COST
            cost, step = min((diagonal, "diag"), (up, "up"), (left, "left"), key=lambda item: item[0])
            costs[row][col] = cost
            steps[row][col] = step

    matches: dict[int, int] = {}
    fuzzy: dict[int, float] = {}
    row = len(canonical)
    col = len(pieces)
    while row or col:
        step = steps[row][col]
        if step == "diag":
            similarity = token_similarity(canonical[row - 1].token, pieces[col - 1].token)
            if similarity >= FUZZY_MATCH_THRESHOLD:
                matches[row - 1] = col - 1
                if similarity < 1.0:
                    fuzzy[row - 1] = similarity
            row -= 1
            col -= 1
            continue
        if step == "up":
            row -= 1
            continue
        col -= 1
    return matches, fuzzy


def substitution_cost(similarity: float) -> float:
    if similarity >= FUZZY_MATCH_THRESHOLD:
        return 1.0 - similarity
    return MISMATCH_COST


def build_lines(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
    canonical: tuple[CanonicalToken, ...],
    pieces: tuple[WordPiece, ...],
    matches: dict[int, int],
    fuzzy: dict[int, float],
) -> tuple[LyricLine, ...]:
    lines: list[LyricLine] = []
    canonical_index = 0
    for line_number, text in enumerate(lyrics, start=1):
        expected = lyric_tokens(text)
        indices = range(canonical_index, canonical_index + len(expected))
        matched_pieces = [matches[index] for index in indices if index in matches]
        unique_piece_indices = unique(matched_pieces)
        line_words = tuple(words[pieces[index].word_index] for index in unique_piece_indices)
        similarities = [
            fuzzy.get(index, 1.0)
            for index in indices
            if index in matches
        ]
        matched_tokens = len(matched_pieces)
        fuzzy_matches = sum(index in fuzzy for index in indices)
        canonical_index += len(expected)
        if not expected:
            lines.append(
                LyricLine(
                    line_number=line_number,
                    text=text,
                    confidence=1.0,
                )
            )
            continue
        if not line_words:
            lines.append(
                LyricLine(
                    line_number=line_number,
                    text=text,
                    expected_tokens=len(expected),
                    matched_tokens=matched_tokens,
                    fuzzy_matches=fuzzy_matches,
                )
            )
            continue
        lines.append(
            LyricLine(
                line_number=line_number,
                text=text,
                words=line_words,
                start=line_words[0].start,
                confidence=line_confidence(len(expected), matched_tokens, line_words, similarities),
                timestamp_source=line_words[0].timestamp_source,
                expected_tokens=len(expected),
                matched_tokens=matched_tokens,
                fuzzy_matches=fuzzy_matches,
            )
        )
    return tuple(lines)


def unique(indices: list[int]) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        values.append(index)
    return tuple(values)


def line_confidence(
    expected_count: int,
    matched_tokens: int,
    words: tuple[AlignedWord, ...],
    similarities: list[float],
) -> float:
    if not expected_count or not words or not similarities:
        return 0.0
    coverage = min(matched_tokens / expected_count, 1.0)
    similarity = sum(similarities) / len(similarities)
    return round(coverage * robust_average_confidence(words) * similarity, 4)


def robust_average_confidence(words: tuple[AlignedWord, ...]) -> float:
    scores = sorted(word.confidence for word in words)
    if len(scores) >= 4:
        scores = scores[1:]
    return sum(scores) / len(scores)


def interpolate_unmatched_lines(
    lines: tuple[LyricLine, ...],
    words: tuple[AlignedWord, ...],
) -> tuple[LyricLine, ...]:
    updated = list(lines)
    index = 0
    while index < len(updated):
        if updated[index].expected_tokens == 0 or updated[index].words:
            index += 1
            continue
        end = index
        while end < len(updated) and updated[end].expected_tokens and not updated[end].words:
            end += 1
        fill_line_starts(updated, index, end, words)
        index = end
    return tuple(updated)


def fill_line_starts(lines: list[LyricLine], start: int, end: int, words: tuple[AlignedWord, ...]) -> None:
    count = end - start
    previous = next((lines[index] for index in range(start - 1, -1, -1) if lines[index].words), None)
    following = next((lines[index] for index in range(end, len(lines)) if lines[index].words), None)
    values = interpolated_values(
        previous.start if previous else None,
        following.start if following else None,
        count,
        words[0].start if words else None,
    )
    source = previous.timestamp_source if previous else following.timestamp_source if following else "interpolated"
    for offset, value in enumerate(values):
        line = lines[start + offset]
        lines[start + offset] = LyricLine(
            line_number=line.line_number,
            text=line.text,
            words=line.words,
            start=value,
            confidence=line.confidence,
            timestamp_source=source or "interpolated",
            expected_tokens=line.expected_tokens,
            matched_tokens=line.matched_tokens,
            fuzzy_matches=line.fuzzy_matches,
            interpolated=True,
        )


def interpolated_values(
    previous: float | None,
    following: float | None,
    count: int,
    default_start: float | None,
) -> tuple[float, ...]:
    if previous is not None and following is not None:
        step = (following - previous) / (count + 1)
        return tuple(previous + step * (index + 1) for index in range(count))
    if previous is not None:
        return tuple(previous + 0.01 * (index + 1) for index in range(count))
    if following is not None:
        base = following - 0.01 * count
        return tuple(max(0.0, base + 0.01 * index) for index in range(count))
    base = default_start or 0.0
    return tuple(base + 0.01 * index for index in range(count))


def timestamp_source_counts(lines: tuple[LyricLine, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        if not line.expected_tokens:
            continue
        counts[line.timestamp_source] = counts.get(line.timestamp_source, 0) + 1
    return counts
