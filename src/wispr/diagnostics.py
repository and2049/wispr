from __future__ import annotations

from pathlib import Path

from wispr.matching import MatchingDiagnostics
from wispr.models import AccuracyDiagnostics, AlignmentSummary, LyricLine, WisprWarning


def build_diagnostics(
    *,
    summary: AlignmentSummary,
    lines: tuple[LyricLine, ...],
    warnings: tuple[WisprWarning, ...],
    fallback_words: int,
    reuse_enabled: bool,
    artifact_hits: tuple[str, ...],
    artifact_misses: tuple[str, ...],
    artifact_invalidations: tuple[str, ...],
    artifact_paths: dict[str, Path],
    matching: MatchingDiagnostics | None = None,
) -> AccuracyDiagnostics:
    lyric_lines = tuple(line for line in lines if line.expected_tokens)
    total_lines = len(lyric_lines)
    total_tokens = sum(line.expected_tokens for line in lyric_lines)
    confidences = tuple(line.confidence for line in lyric_lines)
    matching = matching or MatchingDiagnostics()
    return AccuracyDiagnostics(
        alignment_coverage=summary.alignment_coverage,
        weak_line_ratio=ratio(summary.weak_line_count, total_lines),
        interpolation_ratio=ratio(summary.interpolated_lines, total_lines),
        fuzzy_match_ratio=ratio(summary.fuzzy_matches, total_tokens),
        timestamp_sources=summary.timestamp_sources,
        average_line_confidence=round(sum(confidences) / len(confidences), 4)
        if confidences
        else 0.0,
        lowest_line_confidence=min(confidences) if confidences else 0.0,
        skipped_words=summary.skipped_words,
        fallback_words=fallback_words,
        quality_warnings=quality_warnings(summary, warnings),
        artifact_reuse_enabled=reuse_enabled,
        artifact_hits=len(artifact_hits),
        artifact_misses=len(artifact_misses),
        artifact_invalidations=artifact_invalidations,
        artifact_paths=artifact_paths,
        matching_strategy=matching.strategy,
        matching_fallback_used=matching.fallback_used,
        matching_anchor_count=matching.anchor_count,
        repeated_line_count=matching.repeated_line_count,
        ambiguous_line_count=matching.ambiguous_line_count,
        line_match_scores=matching.line_match_scores,
    )


def ratio(value: int, total: int) -> float:
    return round(value / total, 4) if total else 0.0


def quality_warnings(
    summary: AlignmentSummary,
    warnings: tuple[WisprWarning, ...],
) -> tuple[str, ...]:
    values: list[str] = []
    if summary.quality_warning:
        values.append(summary.quality_warning)
    values.extend(warning.message for warning in warnings)
    return tuple(values)
