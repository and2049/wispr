from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from wispr.backend_factory import BackendName, build_backends
from wispr.models import BatchJob, BatchJobResult, BatchRunResult, TrackMetadata, to_jsonable
from wispr.pipeline import PipelineBackends, run

REQUIRED_COLUMNS = {"audio", "lyrics"}


def read_manifest(path: Path) -> tuple[BatchJob, ...]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Batch manifest does not exist: {manifest_path}.")
    if not manifest_path.is_file():
        raise ValueError(f"Batch manifest is not a file: {manifest_path}.")

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"Batch manifest missing required column(s): {missing_names}.")
        return tuple(
            parse_job(row, index, manifest_path.parent)
            for index, row in enumerate(reader, 2)
        )


def parse_job(row: dict[str, str], row_number: int, base_dir: Path) -> BatchJob:
    return BatchJob(
        row_number=row_number,
        audio_path=manifest_path(base_dir, row["audio"]),
        lyrics_path=manifest_path(base_dir, row["lyrics"]),
        output_path=optional_manifest_path(base_dir, row.get("output")),
        language=optional_text(row.get("language")),
        title=optional_text(row.get("title")),
        artist=optional_text(row.get("artist")),
        album=optional_text(row.get("album")),
    )


def manifest_path(base_dir: Path, value: str) -> Path:
    path = Path(value.strip()).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def optional_manifest_path(base_dir: Path, value: str | None) -> Path | None:
    return manifest_path(base_dir, value) if optional_text(value) else None


def optional_text(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def run_batch(
    manifest_path: Path,
    *,
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
    summary_path: Path | None = None,
    backends_for_language: Callable[[str], PipelineBackends] | None = None,
) -> BatchRunResult:
    start = perf_counter()
    manifest = manifest_path.expanduser().resolve()
    jobs = read_manifest(manifest)
    summary_path = summary_path or manifest.with_suffix(".summary.json")
    backends_for_language = backends_for_language or cached_backend_factory(
        backend=backend,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        demucs=demucs,
    )
    results = tuple(
        run_batch_job(
            job,
            force=force,
            debug=debug,
            demucs=demucs,
            reuse_artifacts=reuse_artifacts,
            default_language=language,
            backends_for_language=backends_for_language,
        )
        for job in jobs
    )
    batch_timings = aggregate_timings(results)
    batch_timings["batch_total"] = perf_counter() - start
    result = BatchRunResult(
        manifest_path=manifest,
        summary_path=summary_path,
        jobs=results,
        total_jobs=len(results),
        succeeded=sum(item.status == "ok" for item in results),
        failed=sum(item.status == "failed" for item in results),
        stage_timings=batch_timings,
    )
    write_batch_summary(result)
    return result


def cached_backend_factory(
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
        return cache[selected_language]

    return backends_for


def run_batch_job(
    job: BatchJob,
    *,
    force: bool,
    debug: bool,
    demucs: bool,
    reuse_artifacts: bool,
    default_language: str,
    backends_for_language: Callable[[str], PipelineBackends],
) -> BatchJobResult:
    try:
        result = run(
            job.audio_path,
            job.lyrics_path,
            output_path=job.output_path,
            force=force,
            debug=debug,
            demucs_enabled=demucs,
            reuse_artifacts=reuse_artifacts,
            metadata_override=job_metadata(job),
            backends=backends_for_language(job.language or default_language),
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        return BatchJobResult(job=job, status="failed", error_message=str(error))
    return BatchJobResult(
        job=job,
        status="ok",
        output_path=result.output_path,
        debug_dir=result.debug_dir,
        summary=result.summary,
        warnings=result.warnings,
        stage_timings=result.stage_timings or {},
        diagnostics=result.diagnostics,
    )


def job_metadata(job: BatchJob) -> TrackMetadata | None:
    if not any((job.title, job.artist, job.album)):
        return None
    return TrackMetadata(title=job.title, artist=job.artist, album=job.album)


def aggregate_timings(results: tuple[BatchJobResult, ...]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for result in results:
        for stage, seconds in result.stage_timings.items():
            totals[stage] = totals.get(stage, 0.0) + seconds
    return totals


def write_batch_summary(result: BatchRunResult) -> None:
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.summary_path.write_text(
        json.dumps(to_jsonable(result), indent=2),
        encoding="utf-8",
    )
