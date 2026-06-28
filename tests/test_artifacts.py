from pathlib import Path

from wispr.artifacts import (
    ArtifactEntry,
    ArtifactManifest,
    file_fingerprint,
    matching_entry,
    read_transcript_artifact,
    transcript_config,
    write_transcript_artifact,
)
from wispr.models import BackendRuntimeConfig, TranscriptWord


def test_file_fingerprint_changes_when_file_changes(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"first")
    first = file_fingerprint(audio)

    audio.write_bytes(b"second")
    second = file_fingerprint(audio)

    assert first != second
    assert first["path"] == second["path"]


def test_manifest_validation_accepts_matching_entry(tmp_path: Path) -> None:
    artifact = tmp_path / "transcript.json"
    artifact.write_text("{}", encoding="utf-8")
    fingerprint = {"path": "song.wav", "size": 4}
    config = transcript_config(BackendRuntimeConfig(backend="whisperx"))
    entry = ArtifactEntry(
        kind="transcript",
        path=artifact,
        fingerprint=fingerprint,
        config=config,
    )

    match, reason = matching_entry(
        ArtifactManifest(tmp_path, {"transcript": entry}),
        "transcript",
        fingerprint=fingerprint,
        config=config,
    )

    assert match == entry
    assert reason is None


def test_manifest_validation_rejects_changed_config(tmp_path: Path) -> None:
    artifact = tmp_path / "transcript.json"
    artifact.write_text("{}", encoding="utf-8")
    fingerprint = {"path": "song.wav", "size": 4}
    entry = ArtifactEntry(
        kind="transcript",
        path=artifact,
        fingerprint=fingerprint,
        config=transcript_config(BackendRuntimeConfig(backend="whisperx", model_name="base")),
    )

    match, reason = matching_entry(
        ArtifactManifest(tmp_path, {"transcript": entry}),
        "transcript",
        fingerprint=fingerprint,
        config=transcript_config(BackendRuntimeConfig(backend="whisperx", model_name="small")),
    )

    assert match is None
    assert reason == "transcript: runtime config changed"


def test_cached_transcript_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "transcript.json"
    transcript = (
        TranscriptWord("hello", start=1.0, end=1.5, confidence=0.8, source="whisperx"),
    )

    write_transcript_artifact(
        path,
        transcript,
        raw={"segments": []},
        skipped_words=2,
        fallback_words=1,
    )

    loaded, skipped, fallback, raw = read_transcript_artifact(path)
    assert loaded == transcript
    assert skipped == 2
    assert fallback == 1
    assert raw == {"segments": []}
