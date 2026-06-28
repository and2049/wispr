from pathlib import Path

import pytest

from wispr.audio import ensure_output_writable, resolve_output_path, validate_audio_path


def test_output_path_defaults_to_audio_stem(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"mock")

    assert resolve_output_path(audio, None) == tmp_path / "song.lrc"


def test_output_path_can_be_overridden(tmp_path: Path) -> None:
    assert resolve_output_path(tmp_path / "song.wav", tmp_path / "out.lrc") == tmp_path / "out.lrc"


def test_output_collision_fails_without_force(tmp_path: Path) -> None:
    output = tmp_path / "song.lrc"
    output.write_text("exists", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ensure_output_writable(output, force=False)


def test_output_collision_allows_force(tmp_path: Path) -> None:
    output = tmp_path / "song.lrc"
    output.write_text("exists", encoding="utf-8")

    ensure_output_writable(output, force=True)


def test_unsupported_audio_extension_fails(tmp_path: Path) -> None:
    audio = tmp_path / "song.ogg"
    audio.write_bytes(b"mock")

    with pytest.raises(ValueError):
        validate_audio_path(audio)


def test_missing_audio_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_audio_path(tmp_path / "missing.wav")


def test_audio_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_audio_path(tmp_path)


def test_valid_audio_path_resolves_absolute(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"mock")

    assert validate_audio_path(audio) == audio.resolve()
