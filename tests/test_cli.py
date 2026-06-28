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


def test_cli_backend_whisperx_wires_backend_factory(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    output = tmp_path / "song.lrc"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")
    calls = {}

    def fake_build_backends(backend, *, model_name: str, device: str, compute_type: str):
        calls["backend"] = backend
        calls["model_name"] = model_name
        calls["device"] = device
        calls["compute_type"] = compute_type
        return None

    def fake_run(audio_path, lyrics_path, **kwargs):
        calls["run_backends"] = kwargs["backends"]
        output.write_text("[00:00.00]hello\n", encoding="utf-8")
        return type("Result", (), {"output_path": output, "warnings": (), "debug_dir": None})()

    monkeypatch.setattr("wispr.cli.build_backends", fake_build_backends)
    monkeypatch.setattr("wispr.cli.run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            str(audio),
            str(lyrics),
            "--backend",
            "whisperx",
            "--model",
            "small",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
        ],
    )

    assert result.exit_code == 0
    assert str(calls["backend"]) == "whisperx"
    assert calls["model_name"] == "small"
    assert calls["device"] == "cpu"
    assert calls["compute_type"] == "int8"
