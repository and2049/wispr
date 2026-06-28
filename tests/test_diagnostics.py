from pathlib import Path

from wispr.diagnostics import build_diagnostics
from wispr.models import AlignmentSummary, LyricLine, WisprWarning


def test_diagnostics_computes_quality_ratios() -> None:
    summary = AlignmentSummary(
        total_lyric_words=4,
        aligned_words=3,
        skipped_words=2,
        average_confidence=0.7,
        weak_line_count=1,
        backend="whisperx",
        alignment_coverage=0.75,
        fuzzy_matches=1,
        interpolated_lines=1,
        timestamp_sources={"whisperx": 1, "interpolated": 1},
        quality_warning="Low lyric coverage",
    )
    lines = (
        LyricLine(1, "one two", expected_tokens=2, confidence=0.8),
        LyricLine(2, "three four", expected_tokens=2, confidence=0.4, interpolated=True),
        LyricLine(3, "", expected_tokens=0, confidence=1.0),
    )
    warnings = (WisprWarning(2, 0.4, "interpolated", "weak line"),)

    diagnostics = build_diagnostics(
        summary=summary,
        lines=lines,
        warnings=warnings,
        fallback_words=1,
        reuse_enabled=True,
        artifact_hits=("transcript",),
        artifact_misses=("demucs",),
        artifact_invalidations=("demucs: runtime config changed",),
        artifact_paths={"transcript": Path("transcript.json")},
    )

    assert diagnostics.weak_line_ratio == 0.5
    assert diagnostics.interpolation_ratio == 0.5
    assert diagnostics.fuzzy_match_ratio == 0.25
    assert diagnostics.average_line_confidence == 0.6
    assert diagnostics.lowest_line_confidence == 0.4
    assert diagnostics.skipped_words == 2
    assert diagnostics.fallback_words == 1
    assert diagnostics.quality_warnings == ("Low lyric coverage", "weak line")
    assert diagnostics.artifact_hits == 1
    assert diagnostics.artifact_misses == 1
