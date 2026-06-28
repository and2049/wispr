from __future__ import annotations

from pathlib import Path
from typing import Annotated

import click
import typer
from typer.core import TyperGroup

from wispr.backend_factory import BackendName, build_backends
from wispr.batch import run_batch as run_batch_manifest
from wispr.benchmark import run_batch_benchmark, run_benchmark
from wispr.models import BatchRunResult, BenchmarkBatchResult, BenchmarkRunResult
from wispr.pipeline import run


class DefaultCommandGroup(TyperGroup):
    def parse_args(self, ctx, args):
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["run", *args]
        return super().parse_args(ctx, args)

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            args.insert(0, "run")
            return super().resolve_command(ctx, args)


app = typer.Typer(cls=DefaultCommandGroup, add_completion=False, no_args_is_help=True)
MAX_WARNING_LINES = 5


@app.command("run")
def run_command(
    audio: Annotated[Path, typer.Argument(help="Input audio file.")],
    lyrics: Annotated[Path, typer.Argument(help="Canonical line-by-line lyrics file.")],
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    demucs: Annotated[bool, typer.Option("--demucs")] = False,
    backend: Annotated[BackendName, typer.Option("--backend")] = BackendName.whisperx,
    model: Annotated[str, typer.Option("--model")] = "base",
    device: Annotated[str, typer.Option("--device")] = "auto",
    compute_type: Annotated[str, typer.Option("--compute-type")] = "auto",
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
    language: Annotated[str, typer.Option("--language")] = "en",
) -> None:
    try:
        backends = build_backends(
            backend,
            model_name=model,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=language,
            demucs=demucs,
        )
        result = run(
            audio,
            lyrics,
            output_path=output,
            force=force,
            debug=debug,
            demucs_enabled=demucs,
            backends=backends,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error

    print_run_result(result)


@app.command("batch")
def batch_command(
    manifest: Annotated[Path, typer.Argument(help="CSV manifest with audio and lyrics columns.")],
    force: Annotated[bool, typer.Option("--force")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    demucs: Annotated[bool, typer.Option("--demucs")] = False,
    backend: Annotated[BackendName, typer.Option("--backend")] = BackendName.whisperx,
    model: Annotated[str, typer.Option("--model")] = "base",
    device: Annotated[str, typer.Option("--device")] = "auto",
    compute_type: Annotated[str, typer.Option("--compute-type")] = "auto",
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
    language: Annotated[str, typer.Option("--language")] = "en",
) -> None:
    try:
        result = run_batch_manifest(
            manifest,
            backend=backend,
            model_name=model,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=language,
            demucs=demucs,
            force=force,
            debug=debug,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error

    print_batch_result(result)


@app.command("benchmark")
def benchmark_command(
    target: Annotated[str, typer.Argument(help="Audio path, or 'batch'.")],
    lyrics_or_manifest: Annotated[Path, typer.Argument(help="Lyrics path, or batch manifest.")],
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
    report: Annotated[Path | None, typer.Option("--report")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    demucs: Annotated[bool, typer.Option("--demucs")] = False,
    backend: Annotated[BackendName, typer.Option("--backend")] = BackendName.whisperx,
    model: Annotated[str, typer.Option("--model")] = "base",
    device: Annotated[str, typer.Option("--device")] = "auto",
    compute_type: Annotated[str, typer.Option("--compute-type")] = "auto",
    batch_size: Annotated[int | None, typer.Option("--batch-size")] = None,
    language: Annotated[str, typer.Option("--language")] = "en",
) -> None:
    try:
        if target == "batch":
            if output is not None:
                raise ValueError("-o/--output is only valid for single-song benchmarks.")
            result = run_batch_benchmark(
                lyrics_or_manifest,
                report_path=report,
                backend=backend,
                model_name=model,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                language=language,
                demucs=demucs,
                force=force,
                debug=debug,
            )
            print_batch_benchmark_result(result)
            return

        result = run_benchmark(
            Path(target),
            lyrics_or_manifest,
            output_path=output,
            report_path=report,
            backend=backend,
            model_name=model,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            language=language,
            demucs=demucs,
            force=force,
            debug=debug,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error

    print_benchmark_result(result)


def print_run_result(result) -> None:
    typer.echo(f"Wrote {result.output_path}")
    typer.echo(
        "Alignment summary: "
        f"backend={result.summary.backend} "
        f"aligned={result.summary.aligned_words}/{result.summary.total_lyric_words} "
        f"skipped={result.summary.skipped_words} "
        f"avg_confidence={result.summary.average_confidence:.2f} "
        f"weak_lines={result.summary.weak_line_count}"
    )
    if result.summary.quality_warning:
        typer.echo(f"warning: {result.summary.quality_warning}", err=True)
    for warning in result.warnings[:MAX_WARNING_LINES]:
        typer.echo(
            f"warning: line {warning.line_number} confidence={warning.confidence:.2f} "
            f"source={warning.timestamp_source}: {warning.message}",
            err=True,
        )
    if result.debug_dir:
        typer.echo(f"Processing audio: {result.processing_audio_path}")
        typer.echo(f"Debug artifacts: {result.debug_dir}")
    remaining_warnings = len(result.warnings) - MAX_WARNING_LINES
    if remaining_warnings > 0:
        typer.echo(f"... {remaining_warnings} more warnings", err=True)


def print_batch_result(result: BatchRunResult) -> None:
    for item in result.jobs:
        if item.status == "ok":
            typer.echo(f"ok row={item.job.row_number} output={item.output_path}")
        else:
            typer.echo(
                f"failed row={item.job.row_number} audio={item.job.audio_path}: "
                f"{item.error_message}",
                err=True,
            )
    typer.echo(
        f"Batch summary: total={result.total_jobs} ok={result.succeeded} "
        f"failed={result.failed} seconds={result.stage_timings.get('batch_total', 0.0):.2f}"
    )
    typer.echo(f"Batch summary file: {result.summary_path}")


def print_benchmark_result(result: BenchmarkRunResult) -> None:
    typer.echo(f"Wrote {result.output_path}")
    typer.echo(f"Benchmark report: {result.report_path}")
    typer.echo(
        "Benchmark summary: "
        f"seconds={result.total_seconds:.2f} "
        f"backend={result.summary.backend} "
        f"aligned={result.summary.aligned_words}/{result.summary.total_lyric_words} "
        f"weak_lines={result.summary.weak_line_count}"
    )
    if result.debug_dir:
        typer.echo(f"Debug artifacts: {result.debug_dir}")


def print_batch_benchmark_result(result: BenchmarkBatchResult) -> None:
    typer.echo(
        f"Benchmark batch: total={result.batch.total_jobs} ok={result.batch.succeeded} "
        f"failed={result.batch.failed} seconds={result.total_seconds:.2f}"
    )
    typer.echo(f"Batch summary file: {result.batch.summary_path}")
    typer.echo(f"Benchmark report: {result.report_path}")
