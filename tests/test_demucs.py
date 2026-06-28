from pathlib import Path
from types import SimpleNamespace

import pytest

from wispr.demucs_backend import (
    INSTALL_MESSAGE,
    DemucsVocalSeparator,
    demucs_command,
    demucs_vocals_path,
    validate_demucs_runtime,
)
from wispr.models import DemucsConfig


def test_demucs_runtime_requires_optional_dependency(monkeypatch) -> None:
    def missing_module(name: str):
        raise ImportError(name)

    monkeypatch.setattr("wispr.demucs_backend.import_module", missing_module)

    with pytest.raises(RuntimeError, match=r"wispr\[separation\]"):
        validate_demucs_runtime()


def test_demucs_command_uses_vocals_two_stem_mode(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    work_dir = tmp_path / "song.demucs"

    command = demucs_command(audio, work_dir, DemucsConfig(device="cpu"))

    assert command[1:7] == ["-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d"]
    assert command[7] == "cpu"
    assert command[-3:] == ["--out", str(work_dir), str(audio)]


def test_demucs_vocal_separator_returns_expected_vocals_path(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    output = tmp_path / "out.lrc"
    vocals = demucs_vocals_path(audio, tmp_path / "out.demucs", DemucsConfig())
    audio.write_bytes(b"mock")
    calls = {}

    def fake_run(command, capture_output, text, check):
        calls["command"] = command
        vocals.parent.mkdir(parents=True)
        vocals.write_bytes(b"vocals")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("wispr.demucs_backend.validate_demucs_runtime", lambda: None)
    monkeypatch.setattr("wispr.demucs_backend.subprocess.run", fake_run)

    separator = DemucsVocalSeparator()
    result = separator.separate(audio, output)

    assert result == vocals
    assert calls["command"][-1] == str(audio)
    assert separator.last_raw_result["returncode"] == 0
    assert separator.last_raw_result["output_path"] == vocals


def test_demucs_vocal_separator_reports_failed_run(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    output = tmp_path / "out.lrc"

    monkeypatch.setattr("wispr.demucs_backend.validate_demucs_runtime", lambda: None)
    monkeypatch.setattr(
        "wispr.demucs_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )

    with pytest.raises(RuntimeError, match="Demucs separation failed: boom"):
        DemucsVocalSeparator().separate(audio, output)


def test_install_message_is_actionable() -> None:
    assert "wispr[separation]" in INSTALL_MESSAGE
