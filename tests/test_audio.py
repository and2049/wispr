from pathlib import Path

import pytest

from wispr.audio import ensure_output_writable, resolve_output_path, validate_audio_path


def test_output_path_defaults_to_audio_stem() -> None:
    assert resolve_output_path(Path("song.wav"), None) == Path("song.lrc")


def test_output_path_can_be_overridden() -> None:
    assert resolve_output_path(Path("song.wav"), Path("out.lrc")) == Path("out.lrc")


def test_output_collision_fails_without_force(tmp_path: Path) -> None:
    output = tmp_path / "song.lrc"
    output.write_text("exists", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ensure_output_writable(output, force=False)


def test_output_collision_allows_force(tmp_path: Path) -> None:
    output = tmp_path / "song.lrc"
    output.write_text("exists", encoding="utf-8")

    ensure_output_writable(output, force=True)


def test_unsupported_audio_extension_fails() -> None:
    with pytest.raises(ValueError):
        validate_audio_path(Path("song.ogg"))
