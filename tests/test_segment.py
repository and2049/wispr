from wispr.models import AlignedWord
from wispr.segment import lyric_tokens, segment_lines, summarize_alignment


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


def test_lyric_tokens_normalize_punctuation_and_apostrophes() -> None:
    assert lyric_tokens("Don't stop, twenty-two!") == ("dont", "stop", "twenty", "two")


def test_segment_lines_preserve_blank_lines_without_warning() -> None:
    lines, warnings = segment_lines(
        ("first line", "", "second line"),
        (
            AlignedWord("first", start=0.0, end=0.2, confidence=0.9),
            AlignedWord("line", start=0.3, end=0.5, confidence=0.9),
            AlignedWord("second", start=1.0, end=1.2, confidence=0.9),
            AlignedWord("line", start=1.3, end=1.5, confidence=0.9),
        ),
    )

    assert lines[1].text == ""
    assert lines[1].start == 0.0
    assert lines[2].start == 1.0
    assert warnings == ()


def test_segment_lines_match_through_extra_backend_words() -> None:
    lines, warnings = segment_lines(
        ("spending my twenties on you",),
        (
            AlignedWord("spending", start=0.0, end=0.2, confidence=0.95),
            AlignedWord("all", start=0.2, end=0.3, confidence=0.7),
            AlignedWord("my", start=0.3, end=0.4, confidence=0.95),
            AlignedWord("twenties", start=0.4, end=0.6, confidence=0.95),
            AlignedWord("on", start=0.6, end=0.7, confidence=0.95),
            AlignedWord("you", start=0.7, end=0.9, confidence=0.95),
        ),
    )

    assert [word.text for word in lines[0].words] == ["spending", "my", "twenties", "on", "you"]
    assert lines[0].start == 0.0
    assert warnings == ()


def test_segment_lines_fall_back_when_words_do_not_match() -> None:
    lines, warnings = segment_lines(
        ("canonical lyric",),
        (AlignedWord("model-text", start=2.0, end=2.5, confidence=0.9),),
    )

    assert lines[0].start == 2.0
    assert lines[0].confidence == 0.45
    assert len(warnings) == 1


def test_segment_line_confidence_uses_average_not_minimum() -> None:
    lines, warnings = segment_lines(
        ("spending my twenties on you",),
        (
            AlignedWord("spending", start=0.0, end=0.2, confidence=0.7),
            AlignedWord("my", start=0.2, end=0.3, confidence=0.0),
            AlignedWord("twenties", start=0.3, end=0.5, confidence=0.7),
            AlignedWord("on", start=0.5, end=0.6, confidence=0.7),
            AlignedWord("you", start=0.6, end=0.8, confidence=0.7),
        ),
    )

    assert lines[0].confidence == 0.7
    assert warnings == ()


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
    assert summary.weak_line_count == 0
    assert summary.backend == "whisperx"
