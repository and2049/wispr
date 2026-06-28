from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from wispr.backend_factory import BackendName, build_backends
from wispr.batch import run_batch
from wispr.models import (
    BackendRuntimeConfig,
    BenchmarkBatchResult,
    BenchmarkRunResult,
    to_jsonable,
)
from wispr.pipeline import PipelineBackends, run


def benchmark_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".benchmark.json")


def batch_benchmark_report_path(manifest_path: Path) -> Path:
    return manifest_path.expanduser().resolve().with_suffix(".benchmark.json")


def run_benchmark(
    audio_path: Path,
    lyrics_path: Path,
    *,
    output_path: Path | None = None,
    report_path: Path | None = None,
    backend: BackendName = BackendName.mock,
    model_name: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int | None = None,
    language: str = "en",
    demucs: bool = False,
    reuse_artifacts: bool = False,
    force: bool = False,
    debug: bool = False,
    backends: PipelineBackends | None = None,
) -> BenchmarkRunResult:
    start = perf_counter()
    backends = backends or build_backends(
        backend,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        demucs=demucs,
    )
    result = run(
        audio_path,
        lyrics_path,
        output_path=output_path,
        force=force,
        debug=debug,
        demucs_enabled=demucs,
        reuse_artifacts=reuse_artifacts,
        backends=backends,
    )
    total_seconds = perf_counter() - start
    selected_report_path = report_path or benchmark_report_path(result.output_path)
    report = BenchmarkRunResult(
        report_path=selected_report_path.expanduser().resolve(),
        command=command_config(
            backend=backend,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=language,
            demucs=demucs,
            reuse_artifacts=reuse_artifacts,
            force=force,
            debug=debug,
        ),
        runtime=backends.runtime,
        output_path=result.output_path,
        debug_dir=result.debug_dir,
        summary=result.summary,
        warnings=result.warnings,
        stage_timings={**(result.stage_timings or {}), "benchmark_total": total_seconds},
        total_seconds=total_seconds,
        diagnostics=result.diagnostics,
    )
    write_report(report.report_path, report)
    return report


def run_batch_benchmark(
    manifest_path: Path,
    *,
    report_path: Path | None = None,
    backend: BackendName = BackendName.mock,
    model_name: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int | None = None,
    language: str = "en",
    demucs: bool = False,
    reuse_artifacts: bool = False,
    force: bool = False,
    debug: bool = False,
) -> BenchmarkBatchResult:
    start = perf_counter()
    runtimes: dict[str, BackendRuntimeConfig] = {}
    batch = run_batch(
        manifest_path,
        backend=backend,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        demucs=demucs,
        reuse_artifacts=reuse_artifacts,
        force=force,
        debug=debug,
        backends_for_language=capturing_backend_factory(
            runtimes,
            backend=backend,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=language,
            demucs=demucs,
        ),
    )
    total_seconds = perf_counter() - start
    selected_report_path = report_path or batch_benchmark_report_path(manifest_path)
    report = BenchmarkBatchResult(
        report_path=selected_report_path.expanduser().resolve(),
        command=command_config(
            backend=backend,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=language,
            demucs=demucs,
            reuse_artifacts=reuse_artifacts,
            force=force,
            debug=debug,
        ),
        runtimes=runtimes,
        batch=batch,
        stage_timings={**batch.stage_timings, "benchmark_total": total_seconds},
        total_seconds=total_seconds,
        diagnostics={
            str(item.job.row_number): item.diagnostics
            for item in batch.jobs
            if item.diagnostics is not None
        },
    )
    write_report(report.report_path, report)
    return report


def capturing_backend_factory(
    runtimes: dict[str, BackendRuntimeConfig],
    *,
    backend: BackendName,
    model_name: str,
    device: str,
    compute_type: str,
    batch_size: int | None,
    language: str,
    demucs: bool,
) -> Callable[[str], PipelineBackends]:
    cache: dict[str, PipelineBackends] = {}
    validated = False

    def backends_for(requested_language: str) -> PipelineBackends:
        nonlocal validated
        selected_language = requested_language or language
        if selected_language not in cache:
            cache[selected_language] = build_backends(
                backend,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                language=selected_language,
                demucs=demucs,
                validate=not validated,
            )
            validated = True
            runtimes[selected_language] = cache[selected_language].runtime
        return cache[selected_language]

    return backends_for


def command_config(
    *,
    backend: BackendName,
    model_name: str,
    device: str,
    compute_type: str,
    batch_size: int | None,
    language: str,
    demucs: bool,
    reuse_artifacts: bool,
    force: bool,
    debug: bool,
) -> dict[str, object]:
    return {
        "backend": backend.value,
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "language": language,
        "demucs": demucs,
        "reuse_artifacts": reuse_artifacts,
        "force": force,
        "debug": debug,
    }


def write_report(path: Path, report: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
