import pytest

from wispr.backend_factory import BackendName, build_backends


def test_mock_backend_runtime_config_uses_cli_values() -> None:
    backends = build_backends(
        BackendName.mock,
        model_name="small",
        device="cpu",
        compute_type="int8",
        language="en",
    )

    assert backends.runtime.backend == "mock"
    assert backends.runtime.model_name == "small"
    assert backends.runtime.device == "cpu"
    assert backends.runtime.compute_type == "int8"
    assert backends.runtime.language == "en"
    assert backends.runtime.demucs_enabled is False
    assert backends.runtime.separator_backend == "none"


def test_mock_backend_rejects_demucs() -> None:
    with pytest.raises(ValueError, match="--demucs requires --backend whisperx"):
        build_backends(BackendName.mock, demucs=True)


def test_whisperx_backend_validates_runtime(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        "wispr.backend_factory.validate_whisperx_runtime",
        lambda: calls.setdefault("validated", True),
    )

    backends = build_backends(BackendName.whisperx, model_name="base")

    assert calls["validated"] is True
    assert backends.runtime.backend == "whisperx"


def test_whisperx_backend_wires_demucs(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        "wispr.backend_factory.validate_whisperx_runtime",
        lambda: calls.setdefault("whisperx", True),
    )
    monkeypatch.setattr(
        "wispr.backend_factory.validate_demucs_runtime",
        lambda: calls.setdefault("demucs", True),
    )

    backends = build_backends(BackendName.whisperx, device="cpu", demucs=True)

    assert calls == {"whisperx": True, "demucs": True}
    assert backends.runtime.demucs_enabled is True
    assert backends.runtime.separator_backend == "demucs"
    assert backends.runtime.separator_model == "htdemucs"
