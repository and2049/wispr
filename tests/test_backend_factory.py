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


def test_whisperx_backend_validates_runtime(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(
        "wispr.backend_factory.validate_whisperx_runtime",
        lambda: calls.setdefault("validated", True),
    )

    backends = build_backends(BackendName.whisperx, model_name="base")

    assert calls["validated"] is True
    assert backends.runtime.backend == "whisperx"
