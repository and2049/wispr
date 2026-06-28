import json
from pathlib import Path

import pytest

from wispr.lyrics import read_lyrics, validate_lyrics_path
from wispr.models import AlignedWord, BackendRuntimeConfig, TranscriptWord
from wispr.pipeline import PipelineBackends, run


def test_lyrics_parsing_preserves_line_text(tmp_path: Path) -> None:
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text(" first line\n\nsecond  line\n", encoding="utf-8")

    assert read_lyrics(lyrics) == (" first line", "", "second  line")


def test_lyrics_validation_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_lyrics_path(tmp_path / "missing.txt")


def test_lyrics_validation_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_lyrics_path(tmp_path)


def test_lyrics_validation_rejects_empty_file(tmp_path: Path) -> None:
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_lyrics_path(lyrics)


def test_pipeline_writes_lrc_and_debug_artifacts(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello world\nnext line\n", encoding="utf-8")

    result = run(audio, lyrics, debug=True)

    assert result.output_path == (tmp_path / "song.lrc").resolve()
    assert result.output_path.read_text(encoding="utf-8").splitlines() == [
        "[00:00.00]hello world",
        "[00:04.00]next line",
    ]
    assert result.warnings == ()
    assert result.debug_dir
    assert (result.debug_dir / "inputs.json").exists()
    assert (result.debug_dir / "transcript.json").exists()
    assert (result.debug_dir / "alignment.json").exists()
    assert (result.debug_dir / "segments.json").exists()

    inputs = json.loads((result.debug_dir / "inputs.json").read_text(encoding="utf-8"))
    assert inputs["inputs"]["audio_path"] == str(audio.resolve())
    assert inputs["inputs"]["lyrics_path"] == str(lyrics.resolve())
    assert inputs["inputs"]["output_path"] == str((tmp_path / "song.lrc").resolve())
    assert inputs["inputs"]["demucs_enabled"] is False
    assert inputs["source_audio_path"] == str(audio.resolve())
    assert inputs["processing_audio_path"] == str(audio.resolve())
    assert inputs["separator"]["raw"] is None
    assert inputs["runtime"]["backend"] == "mock"
    assert inputs["runtime"]["model_name"] == "base"
    assert inputs["runtime"]["device"] == "auto"
    assert inputs["runtime"]["compute_type"] == "auto"
    assert inputs["runtime"]["batch_size"] == 4
    assert inputs["runtime"]["language"] == "en"
    assert inputs["runtime"]["vad_method"] == "silero"
    assert (result.debug_dir / "timings.json").exists()

    transcript = json.loads((result.debug_dir / "transcript.json").read_text(encoding="utf-8"))
    alignment = json.loads((result.debug_dir / "alignment.json").read_text(encoding="utf-8"))
    segments = json.loads((result.debug_dir / "segments.json").read_text(encoding="utf-8"))
    assert transcript == {"raw": None, "normalized": [], "skipped_words": 0, "fallback_words": 0}
    assert alignment["raw"] is None
    assert alignment["fallback_words"] == 0
    assert alignment["normalized"][0]["text"] == "hello"
    assert segments["summary"]["backend"] == "mock"
    assert segments["summary"]["weak_line_count"] == len(result.warnings)
    assert result.summary.backend == "mock"
    assert result.stage_timings
    assert "transcription" in result.stage_timings
    assert (result.debug_dir / "diagnostics.json").exists()
    diagnostics = json.loads((result.debug_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["alignment_coverage"] == 1.0
    assert diagnostics["artifact_reuse_enabled"] is False


def test_pipeline_passes_separated_audio_to_transcriber(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    separated = tmp_path / "vocals.wav"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")
    calls = {}

    class Separator:
        last_raw_result = {"output_path": separated}

        def separate(self, audio_path, output_path):
            calls["separator"] = (audio_path, output_path)
            return separated

    class Transcriber:
        def transcribe(self, audio_path):
            calls["transcriber"] = audio_path
            return ()

    result = run(
        audio,
        lyrics,
        debug=True,
        demucs_enabled=True,
        backends=PipelineBackends(separator=Separator(), transcriber=Transcriber()),
    )

    assert calls["separator"] == (audio.resolve(), (tmp_path / "song.lrc").resolve())
    assert calls["transcriber"] == separated
    assert result.processing_audio_path == separated
    inputs = json.loads((result.debug_dir / "inputs.json").read_text(encoding="utf-8"))
    assert inputs["inputs"]["demucs_enabled"] is True
    assert inputs["processing_audio_path"] == str(separated)
    assert inputs["separator"]["raw"] == {"output_path": str(separated)}


def test_pipeline_reuses_cached_demucs_vocals(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    separated = tmp_path / "generated-vocals.wav"
    audio.write_bytes(b"mock")
    lyrics.write_text("hello\n", encoding="utf-8")
    calls = {"separator": 0}

    class Separator:
        last_raw_result = None

        def separate(self, audio_path, output_path):
            calls["separator"] += 1
            separated.write_bytes(b"vocals")
            return separated

    backends = PipelineBackends(separator=Separator())
    first = run(
        audio,
        lyrics,
        debug=True,
        demucs_enabled=True,
        reuse_artifacts=True,
        backends=backends,
    )

    second = run(
        audio,
        lyrics,
        force=True,
        debug=True,
        demucs_enabled=True,
        reuse_artifacts=True,
        backends=PipelineBackends(separator=Separator()),
    )

    assert calls["separator"] == 1
    assert first.processing_audio_path == tmp_path / "song.artifacts/demucs/vocals.wav"
    assert second.processing_audio_path == first.processing_audio_path
    assert second.diagnostics.artifact_hits == 1


def test_pipeline_reuses_cached_transcript_but_still_aligns(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("canonical lyric\n", encoding="utf-8")
    calls = {"transcriber": 0, "aligner": 0}

    class Transcriber:
        last_raw_result = {"segments": []}
        skipped_words = 0
        fallback_words = 0

        def transcribe(self, audio_path):
            calls["transcriber"] += 1
            return (
                TranscriptWord("canonical", start=2.0, end=2.4, confidence=0.9, source="whisperx"),
                TranscriptWord("lyric", start=2.5, end=2.9, confidence=0.9, source="whisperx"),
            )

    class FailingTranscriber:
        def transcribe(self, audio_path):
            raise AssertionError("transcriber should be skipped")

    class Aligner:
        skipped_words = 0
        fallback_words = 0

        def align(self, transcript, lyric_lines, audio_path=None):
            calls["aligner"] += 1
            return tuple(
                AlignedWord(
                    item.text,
                    start=item.start,
                    end=item.end,
                    confidence=item.confidence,
                    timestamp_source="whisperx",
                )
                for item in transcript
            )

    runtime = BackendRuntimeConfig(backend="whisperx", device="cpu", compute_type="int8")
    run(
        audio,
        lyrics,
        debug=True,
        reuse_artifacts=True,
        backends=PipelineBackends(transcriber=Transcriber(), aligner=Aligner(), runtime=runtime),
    )
    second = run(
        audio,
        lyrics,
        force=True,
        debug=True,
        reuse_artifacts=True,
        backends=PipelineBackends(
            transcriber=FailingTranscriber(),
            aligner=Aligner(),
            runtime=runtime,
        ),
    )

    assert calls["transcriber"] == 1
    assert calls["aligner"] == 2
    assert second.diagnostics.artifact_hits == 1
    assert second.output_path.read_text(encoding="utf-8") == "[00:02.00]canonical lyric\n"


def test_pipeline_preserves_canonical_lyrics_text(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("canonical lyric\n", encoding="utf-8")

    class DifferentTextAligner:
        def align(self, transcript, lyric_lines, audio_path=None):
            return (AlignedWord("model-text", start=2.0, end=2.5, confidence=0.9),)

    result = run(
        audio,
        lyrics,
        backends=PipelineBackends(aligner=DifferentTextAligner()),
    )

    assert result.output_path.read_text(encoding="utf-8") == "[00:02.00]canonical lyric\n"


def test_pipeline_debug_preserves_raw_backend_payloads(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("canonical lyric\n", encoding="utf-8")

    class RawTranscriber:
        last_raw_result = {"segments": [{"text": "heard lyric"}]}
        skipped_words = 2
        fallback_words = 1

        def transcribe(self, audio_path):
            return ()

    class RawAligner:
        last_raw_result = {"segments": [{"words": [{"word": "canonical"}]}]}
        skipped_words = 1
        fallback_words = 2

        def align(self, transcript, lyric_lines, audio_path=None):
            return (AlignedWord("canonical", start=2.0, end=2.5, confidence=0.9),)

    result = run(
        audio,
        lyrics,
        debug=True,
        backends=PipelineBackends(transcriber=RawTranscriber(), aligner=RawAligner()),
    )

    transcript = json.loads((result.debug_dir / "transcript.json").read_text(encoding="utf-8"))
    alignment = json.loads((result.debug_dir / "alignment.json").read_text(encoding="utf-8"))
    segments = json.loads((result.debug_dir / "segments.json").read_text(encoding="utf-8"))

    assert transcript["raw"] == {"segments": [{"text": "heard lyric"}]}
    assert transcript["skipped_words"] == 2
    assert transcript["fallback_words"] == 1
    assert alignment["raw"] == {"segments": [{"words": [{"word": "canonical"}]}]}
    assert alignment["skipped_words"] == 1
    assert alignment["fallback_words"] == 2
    assert segments["summary"]["skipped_words"] == 3


def test_pipeline_rejects_empty_real_backend_transcript(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    lyrics = tmp_path / "lyrics.txt"
    audio.write_bytes(b"mock")
    lyrics.write_text("canonical lyric\n", encoding="utf-8")

    class EmptyTranscriber:
        def transcribe(self, audio_path):
            return ()

    with pytest.raises(ValueError, match="Transcription produced no timed words"):
        run(
            audio,
            lyrics,
            backends=PipelineBackends(
                transcriber=EmptyTranscriber(),
                runtime=PipelineBackends().runtime.__class__(backend="whisperx"),
            ),
        )
