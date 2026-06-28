from __future__ import annotations

from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def validate_audio_path(path: Path) -> Path:
    path = path.expanduser()
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio extension '{path.suffix}'. Supported: {supported}.")
    return path


def default_output_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".lrc")


def resolve_output_path(audio_path: Path, output_path: Path | None) -> Path:
    return output_path.expanduser() if output_path else default_output_path(audio_path)


def ensure_output_writable(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}. Pass --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
