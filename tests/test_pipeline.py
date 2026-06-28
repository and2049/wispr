import json
from pathlib import Path

import pytest

from wispr.lyrics import read_lyrics, validate_lyrics_path
from wispr.models import AlignedWord
from wispr.pipeline import PipelineBackends, run


def test_lyrics_parsing_preserves_line_text(tmp_path: Path) -> None:
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text(" first line\n\nsecond  line\n", encoding="utf-8")

    assert read_lyrics(lyrics) == (" first line", "", "second  line")


def test_lyrics_validation_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_lyrics_path(tmp_path / "missing.txt")


def test_lyrics_validation_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_lyrics_path(tmp_path)


def test_lyrics_validation_rejects_empty_file(tmp_path: Path) -> None:
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_lyrics_path(lyrics)


def test_pipeline_writes_lrc_and_debug_artifacts(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello world\nnext line\n", encoding="utf-8")

    result = run(audio, lyrics, debug=True)

    assert result.output_path == (tmp_path / "song.lrc").resolve()
    assert result.output_path.read_text(encoding="utf-8").splitlines() == [
        "[00:00.00]hello world",
        "[00:04.00]next line",
    ]
    assert result.warnings
    assert result.debug_dir
    assert (result.debug_dir / "inputs.json").exists()
    assert (result.debug_dir / "transcript.json").exists()
    assert (result.debug_dir / "alignment.json").exists()
    assert (result.debug_dir / "segments.json").exists()

    inputs = json.loads((result.debug_dir / "inputs.json").read_text(encoding="utf-8"))
    assert inputs["inputs"]["audio_path"] == str(audio.resolve())
    assert inputs["inputs"]["lyrics_path"] == str(lyrics.resolve())
    assert inputs["inputs"]["output_path"] == str((tmp_path / "song.lrc").resolve())


def test_pipeline_preserves_canonical_lyrics_text(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("canonical lyric\n", encoding="utf-8")

    class DifferentTextAligner:
        def align(self, transcript, lyric_lines, audio_path=None):
            return (AlignedWord("model-text", start=2.0, end=2.5, confidence=0.9),)

    result = run(
        audio,
        lyrics,
        backends=PipelineBackends(aligner=DifferentTextAligner()),
    )

    assert result.output_path.read_text(encoding="utf-8") == "[00:02.00]canonical lyric\n"
