"""Profiles token length distributions for prompts, completions, and combined sequences."""

import numpy as np
from rich.console import Console
from rich.table import Table


def compute_length_percentiles(lengths: list[int]) -> dict[str, float]:
    """Computes P50, P90, P95, P99, and Max from a list of token lengths."""
    if not lengths:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    arr = np.array(lengths)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def display_length_summary(
    prompt_lengths: list[int],
    completion_lengths: list[int],
    total_lengths: list[int],
) -> None:
    """Renders a formatted rich summary table to console."""
    console = Console()
    table = Table(title="Sequence Length Distribution Profiling")

    table.add_column("Sequence Type", style="cyan", no_wrap=True)
    table.add_column("P50", style="magenta")
    table.add_column("P90", style="magenta")
    table.add_column("P95", style="green")
    table.add_column("P99", style="yellow")
    table.add_column("Max", style="red")

    for name, lengths in [
        ("Prompt", prompt_lengths),
        ("Completion", completion_lengths),
        ("Total", total_lengths),
    ]:
        p = compute_length_percentiles(lengths)
        table.add_row(
            name,
            f"{p['p50']:.1f}",
            f"{p['p90']:.1f}",
            f"{p['p95']:.1f}",
            f"{p['p99']:.1f}",
            f"{p['max']:.0f}",
        )

    console.print(table)
