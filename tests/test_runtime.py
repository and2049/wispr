import pytest

from wispr.runtime import default_batch_size, resolve_runtime


def test_runtime_auto_selects_cuda_defaults(monkeypatch) -> None:
    monkeypatch.setattr("wispr.runtime.cuda_available", lambda: (True, None))

    runtime = resolve_runtime(device="auto", compute_type="auto")

    assert runtime.device == "cuda"
    assert runtime.compute_type == "float16"
    assert runtime.batch_size == 16
    assert runtime.note is None


def test_runtime_auto_selects_cpu_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "wispr.runtime.cuda_available",
        lambda: (False, "torch is not installed; falling back to CPU"),
    )

    runtime = resolve_runtime(device="auto", compute_type="auto")

    assert runtime.device == "cpu"
    assert runtime.compute_type == "int8"
    assert runtime.batch_size == 4
    assert "falling back to CPU" in runtime.note


def test_runtime_explicit_cuda_fails_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("wispr.runtime.cuda_available", lambda: (False, "no cuda"))

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_runtime(device="cuda", compute_type="auto")


def test_runtime_keeps_explicit_batch_size(monkeypatch) -> None:
    monkeypatch.setattr("wispr.runtime.cuda_available", lambda: (True, None))

    runtime = resolve_runtime(device="auto", compute_type="auto", batch_size=8)

    assert runtime.batch_size == 8
    assert default_batch_size("cuda") == 16
    assert default_batch_size("cpu") == 4
