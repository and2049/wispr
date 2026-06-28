from wispr.models import AlignedWord
from wispr.segment import segment_lines, summarize_alignment


def test_segment_lines_uses_first_word_timestamp() -> None:
    lines, warnings = segment_lines(
        ("hello world",),
        (
            AlignedWord("hello", start=3.0, end=3.2, confidence=0.9),
            AlignedWord("world", start=3.5, end=3.7, confidence=0.8),
        ),
    )

    assert lines[0].text == "hello world"
    assert lines[0].start == 3.0
    assert warnings == ()


def test_weak_confidence_generates_warning() -> None:
    _, warnings = segment_lines(
        ("hello",),
        (AlignedWord("hello", start=0.0, end=0.2, confidence=0.4, timestamp_source="mock"),),
    )

    assert warnings[0].line_number == 1
    assert warnings[0].confidence == 0.4
    assert warnings[0].timestamp_source == "mock"


def test_alignment_summary_counts_words_and_skips() -> None:
    words = (
        AlignedWord("hello", start=0.0, end=0.2, confidence=0.8),
        AlignedWord("world", start=0.3, end=0.5, confidence=0.6),
    )
    _, warnings = segment_lines(("hello world",), words)

    summary = summarize_alignment(
        ("hello world",),
        words,
        warnings,
        backend="whisperx",
        skipped_words=3,
    )

    assert summary.total_lyric_words == 2
    assert summary.aligned_words == 2
    assert summary.skipped_words == 3
    assert summary.average_confidence == 0.7
    assert summary.weak_line_count == 1
    assert summary.backend == "whisperx"
