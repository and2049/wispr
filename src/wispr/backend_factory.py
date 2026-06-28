from __future__ import annotations

from enum import StrEnum

from wispr.backends import MockAligner, MockTranscriber, NoOpVocalSeparator
from wispr.demucs_backend import DemucsVocalSeparator, validate_demucs_runtime
from wispr.models import AlignmentConfig, BackendRuntimeConfig, DemucsConfig, TranscriptionConfig
from wispr.pipeline import PipelineBackends
from wispr.whisperx_backend import WhisperTranscriber, WhisperXAligner, validate_whisperx_runtime


class BackendName(StrEnum):
    mock = "mock"
    whisperx = "whisperx"


def build_backends(
    backend: BackendName,
    *,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "en",
    demucs: bool = False,
) -> PipelineBackends:
    if demucs and backend is BackendName.mock:
        raise ValueError("--demucs requires --backend whisperx.")
    separator = (
        DemucsVocalSeparator(DemucsConfig(device=device)) if demucs else NoOpVocalSeparator()
    )
    runtime = BackendRuntimeConfig(
        backend=str(backend),
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        demucs_enabled=demucs,
        separator_backend="demucs" if demucs else "none",
        separator_model="htdemucs" if demucs else None,
    )
    if backend is BackendName.mock:
        return PipelineBackends(
            separator=separator,
            transcriber=MockTranscriber(),
            aligner=MockAligner(),
            runtime=runtime,
        )
    validate_whisperx_runtime()
    if demucs:
        validate_demucs_runtime()
    return PipelineBackends(
        separator=separator,
        transcriber=WhisperTranscriber(
            TranscriptionConfig(
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                language=language,
            )
        ),
        aligner=WhisperXAligner(AlignmentConfig(device=device, language=language)),
        runtime=runtime,
    )
