from pathlib import Path

from typer.testing import CliRunner

from wispr.cli import app


def test_cli_writes_lrc(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")

    result = CliRunner().invoke(app, [str(audio), str(lyrics)])

    assert result.exit_code == 0
    assert "Wrote" in result.stdout
    assert (tmp_path / "song.lrc").exists()
