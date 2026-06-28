from __future__ import annotations

from enum import StrEnum

from wispr.backends import MockAligner, MockTranscriber
from wispr.models import AlignmentConfig, BackendRuntimeConfig, TranscriptionConfig
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
) -> PipelineBackends:
    runtime = BackendRuntimeConfig(
        backend=str(backend),
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    if backend is BackendName.mock:
        return PipelineBackends(
            transcriber=MockTranscriber(),
            aligner=MockAligner(),
            runtime=runtime,
        )
    validate_whisperx_runtime()
    return PipelineBackends(
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
