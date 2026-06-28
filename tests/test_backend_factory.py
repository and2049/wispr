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
    assert backends.runtime.batch_size == 4
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

    monkeypatch.setattr(
        "wispr.backend_factory.resolve_runtime",
        lambda **kwargs: type(
            "Runtime",
            (),
            {"device": "cpu", "compute_type": "int8", "batch_size": 4, "note": None},
        )(),
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


def test_backend_factory_records_auto_runtime_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        "wispr.backend_factory.resolve_runtime",
        lambda **kwargs: type(
            "Runtime",
            (),
            {
                "device": "cuda",
                "compute_type": "float16",
                "batch_size": 16,
                "note": None,
            },
        )(),
    )

    backends = build_backends(BackendName.mock, device="auto", compute_type="auto")

    assert backends.runtime.requested_device == "auto"
    assert backends.runtime.requested_compute_type == "auto"
    assert backends.runtime.device == "cuda"
    assert backends.runtime.compute_type == "float16"
    assert backends.runtime.batch_size == 16
