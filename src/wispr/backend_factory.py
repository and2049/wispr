from __future__ import annotations

from enum import StrEnum

from wispr.backends import MockAligner, MockTranscriber, NoOpVocalSeparator
from wispr.demucs_backend import DemucsVocalSeparator, validate_demucs_runtime
from wispr.models import AlignmentConfig, BackendRuntimeConfig, DemucsConfig, TranscriptionConfig
from wispr.pipeline import PipelineBackends
from wispr.runtime import resolve_runtime
from wispr.whisperx_backend import WhisperTranscriber, WhisperXAligner, validate_whisperx_runtime


class BackendName(StrEnum):
    mock = "mock"
    whisperx = "whisperx"


def build_backends(
    backend: BackendName,
    *,
    model_name: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int | None = None,
    language: str = "en",
    demucs: bool = False,
    validate: bool = True,
) -> PipelineBackends:
    if demucs and backend is BackendName.mock:
        raise ValueError("--demucs requires --backend whisperx.")
    runtime_selection = resolve_runtime(
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
    )
    separator = (
        DemucsVocalSeparator(DemucsConfig(device=runtime_selection.device))
        if demucs
        else NoOpVocalSeparator()
    )
    runtime = BackendRuntimeConfig(
        backend=str(backend),
        model_name=model_name,
        device=runtime_selection.device,
        compute_type=runtime_selection.compute_type,
        batch_size=runtime_selection.batch_size,
        language=language,
        demucs_enabled=demucs,
        separator_backend="demucs" if demucs else "none",
        separator_model="htdemucs" if demucs else None,
        requested_device=device,
        requested_compute_type=compute_type,
        runtime_note=runtime_selection.note,
    )
    if backend is BackendName.mock:
        return PipelineBackends(
            separator=separator,
            transcriber=MockTranscriber(),
            aligner=MockAligner(),
            runtime=runtime,
        )
    if validate:
        validate_whisperx_runtime()
    if demucs and validate:
        validate_demucs_runtime()
    return PipelineBackends(
        separator=separator,
        transcriber=WhisperTranscriber(
            TranscriptionConfig(
                model_name=model_name,
                device=runtime_selection.device,
                compute_type=runtime_selection.compute_type,
                batch_size=runtime_selection.batch_size,
                language=language,
            )
        ),
        aligner=WhisperXAligner(
            AlignmentConfig(device=runtime_selection.device, language=language)
        ),
        runtime=runtime,
    )
