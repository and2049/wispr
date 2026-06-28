from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from wispr.cli import app
from wispr.models import AlignmentSummary, BenchmarkBatchResult, BenchmarkRunResult, WisprWarning


def test_cli_writes_lrc(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")

    result = CliRunner().invoke(app, [str(audio), str(lyrics), "--backend", "mock"])

    assert result.exit_code == 0
    assert "Wrote" in result.stdout
    assert "Alignment summary" in result.stdout
    assert (tmp_path / "song.lrc").exists()


def test_cli_backend_whisperx_wires_backend_factory(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    output = tmp_path / "song.lrc"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")
    calls = {}

    def fake_build_backends(
        backend,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        batch_size: int | None,
        language: str,
        demucs: bool,
    ):
        calls["backend"] = backend
        calls["model_name"] = model_name
        calls["device"] = device
        calls["compute_type"] = compute_type
        calls["batch_size"] = batch_size
        calls["language"] = language
        calls["demucs"] = demucs
        return None

    def fake_run(audio_path, lyrics_path, **kwargs):
        calls["run_backends"] = kwargs["backends"]
        calls["demucs_enabled"] = kwargs["demucs_enabled"]
        output.write_text("[00:00.00]hello\n", encoding="utf-8")
        return SimpleNamespace(
            output_path=output,
            warnings=(),
            debug_dir=None,
            processing_audio_path=audio,
            summary=AlignmentSummary(
                total_lyric_words=1,
                aligned_words=1,
                skipped_words=0,
                average_confidence=1.0,
                weak_line_count=0,
                backend="whisperx",
            ),
        )

    monkeypatch.setattr("wispr.cli.build_backends", fake_build_backends)
    monkeypatch.setattr("wispr.cli.run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            str(audio),
            str(lyrics),
            "--model",
            "small",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--batch-size",
            "2",
            "--language",
            "en",
            "--demucs",
        ],
    )

    assert result.exit_code == 0
    assert str(calls["backend"]) == "whisperx"
    assert calls["model_name"] == "small"
    assert calls["device"] == "cpu"
    assert calls["compute_type"] == "int8"
    assert calls["batch_size"] == 2
    assert calls["language"] == "en"
    assert calls["demucs"] is True
    assert calls["demucs_enabled"] is True


def test_cli_truncates_warning_output(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    output = tmp_path / "song.lrc"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")

    def fake_run(audio_path, lyrics_path, **kwargs):
        output.write_text("[00:00.00]hello\n", encoding="utf-8")
        return SimpleNamespace(
            output_path=output,
            debug_dir=tmp_path / "song.debug",
            processing_audio_path=audio,
            warnings=tuple(
                WisprWarning(
                    line_number=index,
                    confidence=0.1,
                    timestamp_source="whisperx",
                    message=f"warning {index}",
                )
                for index in range(1, 8)
            ),
            summary=AlignmentSummary(
                total_lyric_words=1,
                aligned_words=1,
                skipped_words=0,
                average_confidence=1.0,
                weak_line_count=7,
                backend="whisperx",
            ),
        )

    monkeypatch.setattr("wispr.cli.run", fake_run)
    monkeypatch.setattr("wispr.cli.build_backends", lambda *args, **kwargs: None)

    result = CliRunner().invoke(app, [str(audio), str(lyrics)])

    assert result.exit_code == 0
    assert "warning: line 1" in result.stderr
    assert "warning: line 5" in result.stderr
    assert "warning: line 6" not in result.stderr
    assert "... 2 more warnings" in result.stderr


def test_cli_prints_quality_warning(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    output = tmp_path / "song.lrc"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")

    def fake_run(audio_path, lyrics_path, **kwargs):
        output.write_text("[00:00.00]hello\n", encoding="utf-8")
        return SimpleNamespace(
            output_path=output,
            debug_dir=None,
            processing_audio_path=audio,
            warnings=(),
            summary=AlignmentSummary(
                total_lyric_words=4,
                aligned_words=1,
                skipped_words=0,
                average_confidence=0.5,
                weak_line_count=0,
                backend="whisperx",
                quality_warning="Low lyric coverage: matched 1/4 lyric tokens.",
            ),
        )

    monkeypatch.setattr("wispr.cli.run", fake_run)
    monkeypatch.setattr("wispr.cli.build_backends", lambda *args, **kwargs: None)

    result = CliRunner().invoke(app, [str(audio), str(lyrics)])

    assert result.exit_code == 0
    assert "Low lyric coverage" in result.stderr


def test_cli_demucs_with_mock_backend_fails(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [str(audio), str(lyrics), "--backend", "mock", "--demucs"],
    )

    assert result.exit_code != 0
    assert "--demucs requires --backend whisperx" in result.output


def test_cli_batch_wires_batch_runner(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    summary = tmp_path / "manifest.summary.json"
    manifest.write_text("audio,lyrics\nsong.wav,lyrics.txt\n", encoding="utf-8")
    calls = {}

    def fake_run_batch(manifest_path, **kwargs):
        calls["manifest"] = manifest_path
        calls.update(kwargs)
        return SimpleNamespace(
            manifest_path=manifest,
            summary_path=summary,
            total_jobs=1,
            succeeded=1,
            failed=0,
            stage_timings={"batch_total": 0.25},
            jobs=(
                SimpleNamespace(
                    status="ok",
                    output_path=tmp_path / "song.lrc",
                    error_message=None,
                    job=SimpleNamespace(row_number=2, audio_path=tmp_path / "song.wav"),
                ),
            ),
        )

    monkeypatch.setattr("wispr.cli.run_batch_manifest", fake_run_batch)

    result = CliRunner().invoke(
        app,
        [
            "batch",
            str(manifest),
            "--backend",
            "whisperx",
            "--model",
            "small",
            "--device",
            "auto",
            "--compute-type",
            "auto",
            "--batch-size",
            "16",
            "--language",
            "en",
            "--debug",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert calls["manifest"] == manifest
    assert calls["backend"].value == "whisperx"
    assert calls["model_name"] == "small"
    assert calls["device"] == "auto"
    assert calls["compute_type"] == "auto"
    assert calls["batch_size"] == 16
    assert calls["debug"] is True
    assert calls["force"] is True
    assert "Batch summary" in result.stdout


def test_cli_benchmark_mock_writes_report(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["benchmark", str(audio), str(lyrics), "--backend", "mock"],
    )

    assert result.exit_code == 0
    assert "Benchmark report" in result.stdout
    assert (tmp_path / "song.lrc").exists()
    assert (tmp_path / "song.benchmark.json").exists()


def test_cli_batch_benchmark_mock_writes_report(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    manifest = tmp_path / "manifest.csv"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")
    manifest.write_text("audio,lyrics,output\nsong.wav,lyrics.txt,song.lrc\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["benchmark", "batch", str(manifest), "--backend", "mock"],
    )

    assert result.exit_code == 0
    assert "Benchmark report" in result.stdout
    assert (tmp_path / "song.lrc").exists()
    assert (tmp_path / "manifest.benchmark.json").exists()


def test_cli_benchmark_whisperx_wires_runtime_flags(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    output = tmp_path / "song.lrc"
    report = tmp_path / "report.json"
    calls = {}

    def fake_run_benchmark(audio_path, lyrics_path, **kwargs):
        calls["audio_path"] = audio_path
        calls["lyrics_path"] = lyrics_path
        calls.update(kwargs)
        return BenchmarkRunResult(
            report_path=report,
            command={},
            runtime=None,
            output_path=output,
            debug_dir=None,
            summary=AlignmentSummary(
                total_lyric_words=1,
                aligned_words=1,
                skipped_words=0,
                average_confidence=1.0,
                weak_line_count=0,
                backend="whisperx",
            ),
            warnings=(),
            stage_timings={"benchmark_total": 0.1},
            total_seconds=0.1,
        )

    monkeypatch.setattr("wispr.cli.run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            str(audio),
            str(lyrics),
            "--backend",
            "whisperx",
            "--model",
            "small",
            "--device",
            "cuda",
            "--compute-type",
            "float16",
            "--batch-size",
            "16",
            "--language",
            "es",
            "--demucs",
            "--debug",
            "--force",
            "-o",
            str(output),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0
    assert calls["backend"].value == "whisperx"
    assert calls["model_name"] == "small"
    assert calls["device"] == "cuda"
    assert calls["compute_type"] == "float16"
    assert calls["batch_size"] == 16
    assert calls["language"] == "es"
    assert calls["demucs"] is True
    assert calls["debug"] is True
    assert calls["force"] is True
    assert calls["output_path"] == output
    assert calls["report_path"] == report


def test_cli_batch_benchmark_wires_runner(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    report = tmp_path / "manifest.benchmark.json"
    summary = tmp_path / "manifest.summary.json"
    manifest.write_text("audio,lyrics\nsong.wav,lyrics.txt\n", encoding="utf-8")
    calls = {}

    def fake_run_batch_benchmark(manifest_path, **kwargs):
        calls["manifest_path"] = manifest_path
        calls.update(kwargs)
        return BenchmarkBatchResult(
            report_path=report,
            command={},
            runtimes={},
            batch=SimpleNamespace(
                manifest_path=manifest,
                summary_path=summary,
                total_jobs=1,
                succeeded=1,
                failed=0,
                stage_timings={"batch_total": 0.2},
                jobs=(),
            ),
            stage_timings={"benchmark_total": 0.2},
            total_seconds=0.2,
        )

    monkeypatch.setattr("wispr.cli.run_batch_benchmark", fake_run_batch_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "batch",
            str(manifest),
            "--backend",
            "whisperx",
            "--device",
            "auto",
            "--compute-type",
            "auto",
            "--batch-size",
            "8",
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0
    assert calls["manifest_path"] == manifest
    assert calls["backend"].value == "whisperx"
    assert calls["batch_size"] == 8
    assert calls["report_path"] == report
    assert "Benchmark batch" in result.stdout
