from wispr.matching import MatchingConfig, match_lyrics_to_words
from wispr.models import AlignedWord
from wispr.segment import segment_lines


def word(text: str, index: int, confidence: float = 0.9) -> AlignedWord:
    start = float(index)
    return AlignedWord(text, start=start, end=start + 0.2, confidence=confidence)


def test_repeated_chorus_lines_choose_chronological_occurrences() -> None:
    lines, warnings = segment_lines(
        ("stay with me", "verse moves on", "stay with me"),
        tuple(
            word(text, index)
            for index, text in enumerate(
                "stay with me verse moves on stay with me".split(),
                start=1,
            )
        ),
    )

    assert [line.start for line in lines] == [1.0, 4.0, 7.0]
    assert warnings == ()


def test_repeated_identical_lines_use_surrounding_context() -> None:
    lines, warnings = segment_lines(
        ("turn it up", "turn it up", "quiet bridge", "turn it up"),
        tuple(
            word(text, index)
            for index, text in enumerate(
                "turn it up turn it up quiet bridge turn it up".split(),
                start=1,
            )
        ),
    )

    assert [line.start for line in lines] == [1.0, 4.0, 7.0, 9.0]
    assert warnings == ()


def test_backend_adlibs_between_lyric_words_do_not_shift_line_start() -> None:
    lines, warnings = segment_lines(
        ("spending my twenties",),
        (
            word("yeah", 0),
            word("spending", 1),
            word("all", 2),
            word("my", 3),
            word("oh", 4),
            word("twenties", 5),
        ),
    )

    assert lines[0].start == 1.0
    assert [item.text for item in lines[0].words] == ["spending", "my", "twenties"]
    assert warnings == ()


def test_partial_line_match_uses_first_matched_token_timestamp() -> None:
    lines, warnings = segment_lines(
        ("missing word lands here",),
        (
            word("word", 3),
            word("lands", 4),
            word("here", 5),
        ),
    )

    assert lines[0].start == 3.0
    assert lines[0].matched_tokens == 3
    assert warnings == ()


def test_matching_diagnostics_report_repeated_and_ambiguous_lines() -> None:
    result = match_lyrics_to_words(
        ("turn it up", "turn it up"),
        tuple(word(text, index) for index, text in enumerate("turn it up turn it up".split())),
    )

    assert result.diagnostics.strategy == "structured"
    assert result.diagnostics.repeated_line_count == 2
    assert result.diagnostics.ambiguous_line_count >= 1
    assert result.diagnostics.line_match_scores[1] > 0


def test_global_fallback_is_used_when_structured_coverage_is_too_low() -> None:
    result = match_lyrics_to_words(
        ("alpha beta", "gamma delta"),
        tuple(word(text, index) for index, text in enumerate("alpha beta gamma delta".split())),
        MatchingConfig(min_line_coverage=1.1, min_total_coverage=1.1),
    )

    assert result.diagnostics.strategy == "global"
    assert result.diagnostics.fallback_used is True
