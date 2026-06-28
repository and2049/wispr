import csv
import json
from pathlib import Path

import pytest

from wispr.backend_factory import BackendName
from wispr.batch import cached_backend_factory, read_manifest, run_batch
from wispr.models import BackendRuntimeConfig
from wispr.pipeline import PipelineBackends


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_parsing_accepts_optional_columns_and_relative_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "audio": "audio/song.wav",
                "lyrics": "lyrics/song.txt",
                "output": "out/song.lrc",
                "language": "en",
                "title": "Title",
                "artist": "Artist",
                "album": "Album",
            }
        ],
    )

    job = read_manifest(manifest)[0]

    assert job.row_number == 2
    assert job.audio_path == (tmp_path / "audio/song.wav").resolve()
    assert job.lyrics_path == (tmp_path / "lyrics/song.txt").resolve()
    assert job.output_path == (tmp_path / "out/song.lrc").resolve()
    assert job.language == "en"
    assert job.title == "Title"
    assert job.artist == "Artist"
    assert job.album == "Album"


def test_manifest_missing_required_columns_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("audio\nsong.wav\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lyrics"):
        read_manifest(manifest)


def test_batch_mock_writes_multiple_outputs_and_summary(tmp_path: Path) -> None:
    audio1 = tmp_path / "song1.wav"
    audio2 = tmp_path / "song2.wav"
    lyrics1 = tmp_path / "song1.txt"
    lyrics2 = tmp_path / "song2.txt"
    manifest = tmp_path / "manifest.csv"
    audio1.write_bytes(b"mock")
    audio2.write_bytes(b"mock")
    lyrics1.write_text("hello\n", encoding="utf-8")
    lyrics2.write_text("world\n", encoding="utf-8")
    write_manifest(
        manifest,
        [
            {"audio": "song1.wav", "lyrics": "song1.txt", "output": "one.lrc"},
            {"audio": "song2.wav", "lyrics": "song2.txt", "output": "two.lrc"},
        ],
    )

    result = run_batch(manifest, backend=BackendName.mock)

    assert result.total_jobs == 2
    assert result.succeeded == 2
    assert result.failed == 0
    assert (tmp_path / "one.lrc").exists()
    assert (tmp_path / "two.lrc").exists()
    assert result.summary_path == (tmp_path / "manifest.summary.json").resolve()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["succeeded"] == 2
    assert "batch_total" in summary["stage_timings"]


def test_batch_continues_after_row_failure(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "song.txt"
    manifest = tmp_path / "manifest.csv"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")
    write_manifest(
        manifest,
        [
            {"audio": "missing.wav", "lyrics": "song.txt", "output": "bad.lrc"},
            {"audio": "song.wav", "lyrics": "song.txt", "output": "good.lrc"},
        ],
    )

    result = run_batch(manifest, backend=BackendName.mock)

    assert result.total_jobs == 2
    assert result.succeeded == 1
    assert result.failed == 1
    assert result.jobs[0].status == "failed"
    assert result.jobs[1].status == "ok"
    assert (tmp_path / "good.lrc").exists()


def test_batch_uses_language_keyed_backend_cache(monkeypatch) -> None:
    calls = []

    def fake_build_backends(backend, **kwargs):
        calls.append(kwargs)
        return PipelineBackends(runtime=BackendRuntimeConfig(language=kwargs["language"]))

    monkeypatch.setattr("wispr.batch.build_backends", fake_build_backends)

    factory = cached_backend_factory(
        backend=BackendName.mock,
        model_name="base",
        device="cpu",
        compute_type="int8",
        batch_size=None,
        language="en",
        demucs=False,
    )
    factory("en")
    factory("en")
    factory("es")

    assert [call["language"] for call in calls] == ["en", "es"]
    assert [call["validate"] for call in calls] == [True, False]
