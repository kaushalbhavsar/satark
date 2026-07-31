"""Generate sample insider-threat telemetry and run SATARK analysis."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from rich.console import Console

from satark.core.engine import AnalysisEngine
from satark.plugins import create_plugin
from satark.scoring.explainability import why_malicious
from satark.scoring.prioritization import prioritize

console = Console()


def write_sample_csv(path: Path) -> None:
    """Write a small CSV resembling the legacy USB/file feature format."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = ["timestamp,user,host,usb_events,file_reads,file_writes"]
    # Normal baseline
    for i in range(10):
        ts = (start + timedelta(hours=i)).isoformat()
        rows.append(f"{ts},alice,workstation-1,1,5,2")
    # Spike — potential exfiltration
    spike = (start + timedelta(hours=10)).isoformat()
    rows.append(f"{spike},alice,workstation-1,12,80,40")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    """Run the insider plugin against generated sample data."""
    data_path = Path("examples/data/sample_insider.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    write_sample_csv(data_path)

    plugin = create_plugin("insider")
    engine = AnalysisEngine(plugins=[plugin])
    records = []
    import csv

    with data_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    events = engine.ingest_raw("insider", records)
    result = engine.analyze(plugin_name="insider", events=events)
    findings = prioritize(result.findings)

    console.print(f"[green]Processed {result.events_processed} events[/green]")
    for finding in findings:
        console.print(f"\n[bold]{finding.detection.title}[/bold]")
        console.print(why_malicious(finding.detection, finding.score))
        console.print(f"Risk={finding.score.value:.2f} confidence={finding.score.confidence:.2f}")
        for ref in finding.score.references:
            console.print(f"  ref: {ref.source.value}:{ref.identifier} ({ref.name})")


if __name__ == "__main__":
    main()
