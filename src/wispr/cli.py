from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from wispr.pipeline import run

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    audio: Annotated[Path, typer.Argument(help="Input audio file.")],
    lyrics: Annotated[Path, typer.Argument(help="Canonical line-by-line lyrics file.")],
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    separate_vocals: Annotated[bool, typer.Option("--separate-vocals/--no-separate-vocals")] = True,
) -> None:
    try:
        result = run(
            audio,
            lyrics,
            output_path=output,
            force=force,
            debug=debug,
            separate_vocals=separate_vocals,
        )
    except (FileExistsError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(f"Wrote {result.output_path}")
    for warning in result.warnings:
        typer.echo(
            f"warning: line {warning.line_number} confidence={warning.confidence:.2f} "
            f"source={warning.timestamp_source}: {warning.message}",
            err=True,
        )
    if result.debug_dir:
        typer.echo(f"Debug artifacts: {result.debug_dir}")
