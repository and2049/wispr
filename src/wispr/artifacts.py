from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from wispr.models import (
    ArtifactEntry,
    ArtifactManifest,
    ArtifactReuseConfig,
    BackendRuntimeConfig,
    DemucsConfig,
    TranscriptWord,
    to_jsonable,
)

HASH_LIMIT_BYTES = 16 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024


def artifacts_dir(output_path: Path) -> Path:
    return output_path.with_suffix("").with_name(f"{output_path.stem}.artifacts")


def manifest_path(output_path: Path) -> Path:
    return artifacts_dir(output_path) / "manifest.json"


def demucs_artifact_path(output_path: Path) -> Path:
    return artifacts_dir(output_path) / "demucs" / "vocals.wav"


def transcript_artifact_path(output_path: Path) -> Path:
    return artifacts_dir(output_path) / "transcript" / "transcript.json"


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_hash(path, stat.st_size),
    }


def file_hash(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        if size <= HASH_LIMIT_BYTES:
            for chunk in iter(lambda: file.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
            return digest.hexdigest()
        digest.update(file.read(HASH_CHUNK_BYTES))
        file.seek(max(size - HASH_CHUNK_BYTES, 0))
        digest.update(file.read(HASH_CHUNK_BYTES))
    digest.update(str(size).encode())
    return digest.hexdigest()


def load_manifest(output_path: Path) -> ArtifactManifest:
    path = manifest_path(output_path)
    if not path.exists():
        return ArtifactManifest(artifacts_dir=artifacts_dir(output_path))
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = {
        key: ArtifactEntry(
            kind=value["kind"],
            path=Path(value["path"]),
            fingerprint=value["fingerprint"],
            config=value["config"],
        )
        for key, value in data.get("entries", {}).items()
    }
    return ArtifactManifest(artifacts_dir=Path(data["artifacts_dir"]), entries=entries)


def write_manifest(output_path: Path, manifest: ArtifactManifest) -> None:
    path = manifest_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8")
    tmp.replace(path)


def record_entry(
    output_path: Path,
    manifest: ArtifactManifest,
    entry: ArtifactEntry,
) -> ArtifactManifest:
    updated = replace(manifest, entries={**manifest.entries, entry.kind: entry})
    write_manifest(output_path, updated)
    return updated


def matching_entry(
    manifest: ArtifactManifest,
    kind: str,
    *,
    fingerprint: dict[str, Any],
    config: dict[str, Any],
) -> tuple[ArtifactEntry | None, str | None]:
    entry = manifest.entries.get(kind)
    if entry is None:
        return None, f"{kind}: missing artifact manifest entry"
    if entry.fingerprint != fingerprint:
        return None, f"{kind}: source fingerprint changed"
    if entry.config != config:
        return None, f"{kind}: runtime config changed"
    if not entry.path.exists():
        return None, f"{kind}: artifact file missing"
    return entry, None


def demucs_config(runtime: BackendRuntimeConfig) -> dict[str, Any]:
    return {
        "separator_backend": runtime.separator_backend,
        "separator_model": runtime.separator_model or DemucsConfig().model_name,
        "device": runtime.device,
    }


def transcript_config(runtime: BackendRuntimeConfig) -> dict[str, Any]:
    return {
        "backend": runtime.backend,
        "model_name": runtime.model_name,
        "device": runtime.device,
        "compute_type": runtime.compute_type,
        "batch_size": runtime.batch_size,
        "language": runtime.language,
        "vad_method": runtime.vad_method,
    }


def copy_artifact(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def write_transcript_artifact(
    path: Path,
    transcript: tuple[TranscriptWord, ...],
    *,
    raw: object,
    skipped_words: int,
    fallback_words: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            to_jsonable(
                {
                    "raw": raw,
                    "normalized": transcript,
                    "skipped_words": skipped_words,
                    "fallback_words": fallback_words,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def read_transcript_artifact(path: Path) -> tuple[tuple[TranscriptWord, ...], int, int, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    transcript = tuple(
        TranscriptWord(
            text=str(item["text"]),
            start=float(item["start"]),
            end=float(item["end"]),
            confidence=float(item.get("confidence", 1.0)),
            source=str(item.get("source", "artifact")),
        )
        for item in data.get("normalized", [])
    )
    return (
        transcript,
        int(data.get("skipped_words", 0)),
        int(data.get("fallback_words", 0)),
        data.get("raw"),
    )


def enabled_reuse(enabled: bool) -> ArtifactReuseConfig:
    return ArtifactReuseConfig(enabled=enabled)
