"""SATARK command-line interface built with Typer and Rich."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from satark import __version__
from satark.core.config import load_settings
from satark.core.engine import AnalysisEngine
from satark.core.plugin import PluginContext
from satark.plugins import builtin_plugins, create_plugin
from satark.scoring.explainability import why_malicious
from satark.scoring.prioritization import prioritize

app = typer.Typer(
    name="satark",
    help="SATARK — Scalable Automated Technology for Analysis and Ranking of Known Threats",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """SATARK security analytics CLI."""


@app.command()
def version() -> None:
    """Show the SATARK version."""
    console.print(f"satark {__version__}")


@app.command("list-plugins")
def list_plugins() -> None:
    """List built-in analytics plugins."""
    table = Table(title="Built-in Plugins")
    table.add_column("Name")
    table.add_column("Domain")
    table.add_column("Description")
    for name in builtin_plugins():
        plugin = create_plugin(name)
        table.add_row(plugin.meta.name, plugin.meta.domain, plugin.meta.description)
    console.print(table)


@app.command()
def analyze(
    plugin: Annotated[str, typer.Option("--plugin", "-p", help="Plugin to run")] = "insider",
    data: Annotated[
        Path | None,
        typer.Option("--data", "-d", help="CSV or JSONL of raw records"),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Elevated risk threshold"),
    ] = 0.7,
    explain: Annotated[bool, typer.Option("--explain", help="Print why-malicious answers")] = True,
) -> None:
    """Run analysis with a plugin against optional input data."""
    settings = load_settings(risk_threshold=threshold)
    engine = AnalysisEngine(plugins=[create_plugin(plugin)], settings=settings)

    if data is None:
        console.print("[yellow]No data provided; showing plugin metadata only.[/yellow]")
        meta = engine.get_plugin(plugin).meta
        console.print(f"{meta.name} v{meta.version} — {meta.description}")
        raise typer.Exit(code=0)

    records = _load_records(data)
    context = PluginContext(config={"data_path": str(data)})
    events = engine.ingest_raw(plugin, records, context)
    result = engine.analyze(plugin_name=plugin, events=events, context=context)
    findings = prioritize(result.findings)

    table = Table(title=f"Findings ({plugin})")
    table.add_column("Title")
    table.add_column("Severity")
    table.add_column("Risk")
    table.add_column("Confidence")
    for finding in findings:
        table.add_row(
            finding.detection.title,
            finding.detection.severity.value,
            f"{finding.score.value:.2f}",
            f"{finding.score.confidence:.2f}",
        )
    console.print(table)
    console.print(
        f"Processed {result.events_processed} events; "
        f"{len(result.elevated)} elevated (threshold={threshold})."
    )

    if explain:
        for finding in findings:
            console.print(f"\n[bold]{finding.detection.title}[/bold]")
            console.print(why_malicious(finding.detection, finding.score))
            console.print(finding.explanation)


@app.command()
def explain_finding(
    path: Annotated[Path, typer.Argument(help="JSON file containing a Finding dump")],
) -> None:
    """Print an explainability report for a serialized finding."""
    from satark.core.models.finding import Finding

    finding = Finding.model_validate_json(path.read_text(encoding="utf-8"))
    console.print(why_malicious(finding.detection, finding.score))
    console.print(finding.explanation)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        msg = "JSON input must be a list of records"
        raise typer.BadParameter(msg)
    # Default: CSV
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    app()
