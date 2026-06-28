from pathlib import Path

import pytest
from typer.testing import CliRunner

from wispr.cli import app


@pytest.mark.ml
def test_local_whisperx_smoke_writes_lrc_and_debug(request: pytest.FixtureRequest) -> None:
    if request.config.option.markexpr != "ml":
        pytest.skip("Run explicitly with: uv run pytest -m ml")

    root = Path(__file__).resolve().parents[1]
    audio = root / "inputs" / "03-giveon-twenties.flac"
    lyrics = root / "inputs" / "lyrics.txt"
    if not audio.exists() or not lyrics.exists():
        pytest.skip("Local inputs/ smoke files are not present.")

    output = root / "inputs" / "out" / "twenties.lrc"
    result = CliRunner().invoke(
        app,
        [
            str(audio),
            str(lyrics),
            "--backend",
            "whisperx",
            "--model",
            "base",
            "--device",
            "cpu",
            "--compute-type",
            "int8",
            "--debug",
            "--force",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    debug_dir = output.with_suffix("").with_name(f"{output.stem}.debug")
    assert (debug_dir / "inputs.json").exists()
    assert (debug_dir / "transcript.json").exists()
    assert (debug_dir / "alignment.json").exists()
    assert (debug_dir / "segments.json").exists()
