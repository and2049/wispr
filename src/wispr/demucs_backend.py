from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path

from wispr.models import DemucsConfig

INSTALL_MESSAGE = (
    "Demucs separation requires optional dependencies. Install with: wispr-lrc[separation]"
)


def validate_demucs_runtime() -> None:
    try:
        import_module("demucs")
    except ImportError as error:
        raise RuntimeError(INSTALL_MESSAGE) from error


class DemucsVocalSeparator:
    def __init__(self, config: DemucsConfig | None = None) -> None:
        self.config = config or DemucsConfig()
        self.last_raw_result: dict[str, object] | None = None

    def separate(self, audio_path: Path, output_path: Path) -> Path:
        validate_demucs_runtime()
        work_dir = output_path.with_suffix("").with_name(f"{output_path.stem}.demucs")
        command = demucs_command(audio_path, work_dir, self.config)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        vocals = demucs_vocals_path(audio_path, work_dir, self.config)
        self.last_raw_result = {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_path": vocals,
        }
        if result.returncode != 0:
            raise RuntimeError(f"Demucs separation failed: {result.stderr.strip()}")
        if not vocals.exists():
            raise RuntimeError(f"Demucs did not produce expected vocals file: {vocals}.")
        return vocals


def demucs_command(audio_path: Path, work_dir: Path, config: DemucsConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        config.model_name,
        "-d",
        config.device,
        "--out",
        str(work_dir),
        str(audio_path),
    ]


def demucs_vocals_path(audio_path: Path, work_dir: Path, config: DemucsConfig) -> Path:
    return work_dir / config.model_name / audio_path.stem / "vocals.wav"
