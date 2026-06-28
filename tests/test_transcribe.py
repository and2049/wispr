from pathlib import Path
from types import SimpleNamespace

import pytest

from wispr.models import AlignmentConfig, TranscriptWord
from wispr.transcribe import TranscriptionConfig, WhisperTranscriber, WhisperXAligner
from wispr.whisperx_backend import (
    FFMPEG_MESSAGE,
    TORCHCODEC_WARNING,
    aligned_words,
    aligned_words_with_skips,
    canonical_segments,
    quiet_stdout,
    quiet_streams,
    suppress_runtime_warnings,
    transcript_words,
    transcript_words_with_skips,
    validate_whisperx_runtime,
)


def test_transcription_config_defaults_are_cpu_safe() -> None:
    config = TranscriptionConfig()

    assert config.model_name == "base"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.batch_size == 4
    assert config.language == "en"
    assert config.vad_method == "silero"


def test_alignment_config_defaults_are_cpu_safe() -> None:
    config = AlignmentConfig()

    assert config.device == "cpu"
    assert config.language == "en"
    assert config.return_char_alignments is False


def test_whisper_transcriber_requires_optional_dependency(monkeypatch, tmp_path: Path) -> None:
    transcriber = WhisperTranscriber(TranscriptionConfig(model_name="base"))

    def missing_module(name: str):
        raise ImportError(name)

    monkeypatch.setattr("wispr.whisperx_backend.import_module", missing_module)

    with pytest.raises(RuntimeError, match=r"wispr\[ml\]"):
        transcriber.transcribe(tmp_path / "song.wav")


def test_whisperx_segments_convert_to_transcript_words() -> None:
    words = transcript_words(
        {
            "segments": [
                {"words": [{"word": " hello ", "start": 1, "end": 1.5, "score": 0.8}]},
            ]
        }
    )

    assert words[0].text == "hello"
    assert words[0].start == 1.0
    assert words[0].end == 1.5
    assert words[0].confidence == 0.8
    assert words[0].source == "whisperx"


def test_whisperx_transcript_splits_merged_words() -> None:
    words = transcript_words(
        {
            "segments": [
                {"words": [{"word": "know\nThis", "start": 1.0, "end": 3.0, "score": 0.8}]},
            ]
        }
    )

    assert [word.text for word in words] == ["know", "This"]
    assert words[0].start == 1.0
    assert words[0].end == 2.0
    assert words[1].start == 2.0
    assert words[1].end == 3.0


def test_whisperx_transcript_conversion_counts_skipped_words() -> None:
    words, skipped, fallback = transcript_words_with_skips(
        {
            "segments": [
                {
                    "words": [
                        {"word": "hello", "start": 1, "end": 1.5},
                        {"word": "missing-start", "end": 2},
                        {"word": "", "start": 2, "end": 3},
                    ]
                },
            ]
        }
    )

    assert len(words) == 1
    assert skipped == 2
    assert fallback == 0


def test_whisperx_transcript_falls_back_to_segment_timing() -> None:
    words, skipped, fallback = transcript_words_with_skips(
        {
            "segments": [
                {"text": "heard lyric", "start": 1.25, "end": 2.75, "avg_logprob": -0.5},
            ]
        }
    )

    assert skipped == 0
    assert fallback == 0
    assert len(words) == 1
    assert words[0].text == "heard lyric"
    assert words[0].start == 1.25
    assert words[0].end == 2.75
    assert round(words[0].confidence, 4) == 0.6065
    assert words[0].source == "whisperx-segment"


def test_whisperx_segments_convert_to_aligned_words() -> None:
    words = aligned_words(
        {
            "segments": [
                {"words": [{"word": "hello", "start": 2, "end": 2.25, "score": 0.9}]},
            ]
        }
    )

    assert words[0].text == "hello"
    assert words[0].start == 2.0
    assert words[0].end == 2.25
    assert words[0].confidence == 0.9
    assert words[0].timestamp_source == "whisperx"


def test_whisperx_alignment_splits_merged_words() -> None:
    words = aligned_words(
        {
            "segments": [
                {"words": [{"word": "it\nDon't", "start": 2.0, "end": 4.0, "score": 0.9}]},
            ]
        }
    )

    assert [word.text for word in words] == ["it", "Don't"]
    assert words[0].start == 2.0
    assert words[0].end == 3.0
    assert words[1].start == 3.0
    assert words[1].end == 4.0


def test_whisperx_alignment_conversion_counts_skipped_words() -> None:
    words, skipped, fallback = aligned_words_with_skips(
        {
            "segments": [
                {
                    "words": [
                        {"word": "hello", "start": 2, "end": 2.25},
                        {"word": "missing-end", "start": 3},
                    ]
                },
            ]
        }
    )

    assert len(words) == 1
    assert skipped == 1
    assert fallback == 0


def test_whisperx_alignment_falls_back_to_segment_timing() -> None:
    words, skipped, fallback = aligned_words_with_skips(
        {
            "segments": [
                {
                    "start": 3.0,
                    "end": 4.0,
                    "words": [{"word": "hello"}],
                },
            ]
        }
    )

    assert skipped == 0
    assert fallback == 1
    assert words[0].start == 3.0
    assert words[0].end == 4.0
    assert words[0].timestamp_source == "whisperx-segment"


def test_whisperx_runtime_requires_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr("wispr.whisperx_backend.import_whisperx", lambda: object())
    monkeypatch.setattr("wispr.whisperx_backend.which", lambda name: None)

    with pytest.raises(RuntimeError, match=FFMPEG_MESSAGE):
        validate_whisperx_runtime()


def test_suppress_runtime_warnings_configures_filter(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        "wispr.whisperx_backend.warnings.filterwarnings",
        lambda action, message, category: calls.update(
            {"action": action, "message": message, "category": category}
        ),
    )

    suppress_runtime_warnings()

    assert calls == {"action": "ignore", "message": TORCHCODEC_WARNING, "category": UserWarning}


def test_quiet_stdout_returns_buffer() -> None:
    buffer = quiet_stdout()
    buffer.write("hidden")

    assert buffer.getvalue() == "hidden"


def test_quiet_streams_return_buffers() -> None:
    stdout, stderr = quiet_streams()
    stdout.write("out")
    stderr.write("err")

    assert stdout.getvalue() == "out"
    assert stderr.getvalue() == "err"


def test_canonical_segments_use_transcript_time_bounds() -> None:
    segments = canonical_segments(
        ("canonical", "lyrics"),
        (
            TranscriptWord("model", start=1.0, end=1.5),
            TranscriptWord("words", start=2.0, end=3.0),
        ),
    )

    assert segments == [{"text": "canonical\nlyrics", "start": 1.0, "end": 3.0}]


def test_canonical_segments_split_when_using_segment_fallback() -> None:
    segments = canonical_segments(
        ("line one", "line two", "line three"),
        (
            TranscriptWord("first segment words", start=1.0, end=2.0, source="whisperx-segment"),
            TranscriptWord("second segment", start=2.0, end=3.0, source="whisperx-segment"),
        ),
    )

    assert segments == [
        {"text": "line one\nline two", "start": 1.0, "end": 2.0},
        {"text": "line three", "start": 2.0, "end": 3.0},
    ]


def test_whisperx_transcriber_calls_backend_with_config(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "song.wav"
    calls = {}

    class FakeModel:
        def transcribe(self, audio, batch_size: int):
            calls["transcribe"] = (audio, batch_size)
            return {"segments": [{"words": [{"word": "hello", "start": 0, "end": 1}]}]}

    fake_whisperx = SimpleNamespace(
        load_model=lambda model, device, compute_type, language, vad_method: calls.setdefault(
            "load_model", (model, device, compute_type, language, vad_method)
        )
        or FakeModel(),
        load_audio=lambda path: calls.setdefault("load_audio", path) or "audio",
    )
    fake_whisperx.load_model = lambda model, device, compute_type, language, vad_method: (
        calls.setdefault("load_model", (model, device, compute_type, language, vad_method)),
        FakeModel(),
    )[1]
    fake_whisperx.load_audio = lambda path: (calls.setdefault("load_audio", path), "audio")[1]
    monkeypatch.setattr("wispr.whisperx_backend.import_whisperx", lambda: fake_whisperx)

    words = WhisperTranscriber(
        TranscriptionConfig(model_name="small", device="cpu", compute_type="int8", batch_size=2)
    ).transcribe(audio_path)

    assert calls["load_model"] == ("small", "cpu", "int8", "en", "silero")
    assert calls["load_audio"] == str(audio_path)
    assert calls["transcribe"] == ("audio", 2)
    assert words[0].text == "hello"
    assert words[0].source == "whisperx"


def test_whisperx_aligner_calls_backend_with_canonical_lyrics(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "song.wav"
    calls = {}

    fake_whisperx = SimpleNamespace()
    fake_whisperx.load_align_model = lambda language_code, device: (
        calls.setdefault("load_align_model", (language_code, device)),
        ("model", "metadata"),
    )[1]
    fake_whisperx.load_audio = lambda path: (calls.setdefault("load_audio", path), "audio")[1]

    def fake_align(segments, model, metadata, audio, device, return_char_alignments):
        calls["align"] = (segments, model, metadata, audio, device, return_char_alignments)
        return {"segments": [{"words": [{"word": "canonical", "start": 0, "end": 1}]}]}

    fake_whisperx.align = fake_align
    monkeypatch.setattr("wispr.whisperx_backend.import_whisperx", lambda: fake_whisperx)

    transcript = (TranscriptWord("heard", start=1.0, end=3.0),)
    words = WhisperXAligner().align(transcript, ("canonical line",), audio_path)

    assert calls["load_align_model"] == ("en", "cpu")
    assert calls["load_audio"] == str(audio_path)
    assert calls["align"][0] == [{"text": "canonical line", "start": 1.0, "end": 3.0}]
    assert calls["align"][3] == "audio"
    assert words[0].text == "canonical"


def test_whisperx_aligner_falls_back_to_raw_transcript(monkeypatch, tmp_path: Path) -> None:
    fake_whisperx = SimpleNamespace(
        load_align_model=lambda language_code, device: ("model", "metadata"),
        load_audio=lambda path: "audio",
        align=lambda *args, **kwargs: {"segments": [{"words": [{"word": "missing"}]}]},
    )
    monkeypatch.setattr("wispr.whisperx_backend.import_whisperx", lambda: fake_whisperx)

    words = WhisperXAligner().align(
        (TranscriptWord("heard", start=1.0, end=2.0, confidence=0.7),),
        ("canonical",),
        tmp_path / "song.wav",
    )

    assert words[0].text == "heard"
    assert words[0].timestamp_source == "raw-whisper"
