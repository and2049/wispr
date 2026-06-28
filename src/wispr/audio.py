from __future__ import annotations

from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def validate_audio_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}.")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}.")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio extension '{path.suffix}'. Supported: {supported}.")
    return path


def default_output_path(audio_path: Path) -> Path:
    return audio_path.expanduser().resolve().with_suffix(".lrc")


def resolve_output_path(audio_path: Path, output_path: Path | None) -> Path:
    return output_path.expanduser().resolve() if output_path else default_output_path(audio_path)


def ensure_output_writable(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}. Pass --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
