from pathlib import Path

from wispr.lyrics import read_lyrics
from wispr.pipeline import run


def test_lyrics_parsing_preserves_line_text(tmp_path: Path) -> None:
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text(" first line\nsecond  line\n", encoding="utf-8")

    assert read_lyrics(lyrics) == (" first line", "second  line")


def test_pipeline_writes_lrc_and_debug_artifacts(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello world\nnext line\n", encoding="utf-8")

    result = run(audio, lyrics, debug=True)

    assert result.output_path == tmp_path / "song.lrc"
    assert result.output_path.read_text(encoding="utf-8").splitlines() == [
        "[00:00.00]hello world",
        "[00:04.00]next line",
    ]
    assert result.warnings
    assert result.debug_dir
    assert (result.debug_dir / "transcript.json").exists()
    assert (result.debug_dir / "alignment.json").exists()
    assert (result.debug_dir / "segments.json").exists()
