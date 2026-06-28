from __future__ import annotations

from wispr.matching import (
    MatchingDiagnostics,
    aligned_word_pieces,
    lyric_tokens,
    match_lyrics_to_words,
)
from wispr.models import AlignedWord, AlignmentSummary, LyricLine, WisprWarning

WEAK_CONFIDENCE_THRESHOLD = 0.55


def segment_lines(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
) -> tuple[tuple[LyricLine, ...], tuple[WisprWarning, ...]]:
    lines, warnings, _ = segment_lines_with_diagnostics(lyrics, words)
    return lines, warnings


def segment_lines_with_diagnostics(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
) -> tuple[tuple[LyricLine, ...], tuple[WisprWarning, ...], MatchingDiagnostics]:
    matching = match_lyrics_to_words(lyrics, words)
    lines = interpolate_unmatched_lines(
        build_lines(lyrics, words, matching.line_matches),
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
    return lines, warnings, matching.diagnostics


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


def build_lines(
    lyrics: tuple[str, ...],
    words: tuple[AlignedWord, ...],
    line_matches,
) -> tuple[LyricLine, ...]:
    pieces = aligned_word_pieces(words)
    lines: list[LyricLine] = []
    for line_number, (text, match) in enumerate(zip(lyrics, line_matches, strict=True), start=1):
        expected = lyric_tokens(text)
        unique_piece_indices = unique([item.piece_index for item in match.matches])
        line_words = tuple(words[pieces[index].word_index] for index in unique_piece_indices)
        similarities = [item.similarity for item in match.matches]
        fuzzy_matches = sum(item.similarity < 1.0 for item in match.matches)
        if not expected:
            lines.append(LyricLine(line_number=line_number, text=text, confidence=1.0))
            continue
        if not line_words:
            lines.append(
                LyricLine(
                    line_number=line_number,
                    text=text,
                    expected_tokens=len(expected),
                    matched_tokens=len(match.matches),
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
                confidence=line_confidence(
                    len(expected),
                    len(match.matches),
                    line_words,
                    similarities,
                ),
                timestamp_source=line_words[0].timestamp_source,
                expected_tokens=len(expected),
                matched_tokens=len(match.matches),
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


def fill_line_starts(
    lines: list[LyricLine],
    start: int,
    end: int,
    words: tuple[AlignedWord, ...],
) -> None:
    count = end - start
    previous = next(
        (lines[index] for index in range(start - 1, -1, -1) if lines[index].words),
        None,
    )
    following = next((lines[index] for index in range(end, len(lines)) if lines[index].words), None)
    values = interpolated_values(
        previous.start if previous else None,
        following.start if following else None,
        count,
        words[0].start if words else None,
    )
    source = (
        previous.timestamp_source
        if previous
        else following.timestamp_source
        if following
        else "interpolated"
    )
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
