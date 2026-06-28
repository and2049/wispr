from pathlib import Path

import pytest

from wispr.transcribe import TranscriptionConfig, WhisperTranscriber


def test_whisper_transcriber_is_deferred(tmp_path: Path) -> None:
    transcriber = WhisperTranscriber(TranscriptionConfig(model_name="base"))

    with pytest.raises(NotImplementedError, match="intentionally deferred"):
        transcriber.transcribe(tmp_path / "song.wav")
