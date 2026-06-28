import json
from pathlib import Path

from wispr.backend_factory import BackendName
from wispr.benchmark import (
    batch_benchmark_report_path,
    benchmark_report_path,
    run_batch_benchmark,
    run_benchmark,
)


def test_benchmark_mock_writes_lrc_and_report(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello world\n", encoding="utf-8")

    result = run_benchmark(audio, lyrics, backend=BackendName.mock)

    assert result.output_path == tmp_path / "song.lrc"
    assert result.report_path == tmp_path / "song.benchmark.json"
    assert result.total_seconds > 0
    assert "transcription" in result.stage_timings
    assert "benchmark_total" in result.stage_timings
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["runtime"]["backend"] == "mock"
    assert report["command"]["backend"] == "mock"
    assert report["summary"]["backend"] == "mock"
    assert report["output_path"] == str(tmp_path / "song.lrc")


def test_benchmark_default_report_paths_are_deterministic(tmp_path: Path) -> None:
    assert benchmark_report_path(tmp_path / "song.lrc") == tmp_path / "song.benchmark.json"
    assert batch_benchmark_report_path(tmp_path / "manifest.csv") == (
        tmp_path / "manifest.benchmark.json"
    ).resolve()


def test_batch_benchmark_mock_writes_outputs_and_report(tmp_path: Path) -> None:
    audio1 = tmp_path / "song1.wav"
    audio2 = tmp_path / "song2.wav"
    lyrics1 = tmp_path / "song1.txt"
    lyrics2 = tmp_path / "song2.txt"
    manifest = tmp_path / "manifest.csv"
    audio1.write_bytes(b"mock")
    audio2.write_bytes(b"mock")
    lyrics1.write_text("hello\n", encoding="utf-8")
    lyrics2.write_text("world\n", encoding="utf-8")
    manifest.write_text(
        "audio,lyrics,output\nsong1.wav,song1.txt,one.lrc\nsong2.wav,song2.txt,two.lrc\n",
        encoding="utf-8",
    )

    result = run_batch_benchmark(manifest, backend=BackendName.mock)

    assert result.report_path == tmp_path / "manifest.benchmark.json"
    assert result.batch.total_jobs == 2
    assert result.batch.succeeded == 2
    assert result.total_seconds > 0
    assert (tmp_path / "one.lrc").exists()
    assert (tmp_path / "two.lrc").exists()
    assert "batch_total" in result.stage_timings
    assert "benchmark_total" in result.stage_timings
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["batch"]["succeeded"] == 2
    assert report["runtimes"]["en"]["backend"] == "mock"
