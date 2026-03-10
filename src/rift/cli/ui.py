from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

T = TypeVar("T")

console = Console()


def rich_enabled() -> bool:
    return console.is_terminal


def emit_json(payload: Any) -> None:
    if rich_enabled():
        console.print(JSON.from_data(payload))
        return
    print(json.dumps(payload, indent=2, default=str))


def emit_markdown(markdown: str) -> None:
    if rich_enabled():
        console.print(Markdown(markdown))
        return
    print(markdown)


def emit_syntax(payload: Any) -> None:
    if rich_enabled():
        console.print(Syntax(json.dumps(payload, indent=2, default=str), "json", theme="monokai", line_numbers=False))
        return
    print(json.dumps(payload, indent=2, default=str))


def emit_panel(title: str, body: str, style: str = "cyan") -> None:
    if rich_enabled():
        console.print(Panel.fit(body, title=title, border_style=style))
        return
    print(f"{title}\n{body}")


def _status_style(value: Any) -> tuple[str, str]:
    text = str(value).lower()
    if value is True or "ok" in text or "healthy" in text or "complete" in text:
        return ("[green]✓[/green]", "green")
    if value is False or "severe" in text or "error" in text or "drift" in text or "bad" in text:
        return ("[red]🚨[/red]", "red")
    if "warn" in text or "review" in text or "pending" in text:
        return ("[yellow]⚠[/yellow]", "yellow")
    return ("[cyan]•[/cyan]", "cyan")


def emit_records(title: str, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    if not rows:
        emit_panel(title, "No records found.", "yellow")
        return

    ordered_columns = list(columns or rows[0].keys())
    if rich_enabled():
        table = Table(title=title, border_style="bright_blue", header_style="bold cyan")
        for column in ordered_columns:
            table.add_column(column, overflow="fold")
        for row in rows:
            rendered = []
            for column in ordered_columns:
                value = row.get(column, "")
                if isinstance(value, bool):
                    badge, _ = _status_style(value)
                    rendered.append(badge)
                else:
                    badge, style = _status_style(value)
                    if column.startswith("is_") or column.endswith("status") or column.endswith("triggered") or column == "decision":
                        rendered.append(f"[{style}]{value}[/{style}]" if style != "cyan" else str(value))
                    else:
                        rendered.append(str(value))
            table.add_row(*rendered)
        console.print(table)
        return

    print(json.dumps(list(rows), indent=2, default=str))


def emit_summary(title: str, items: Sequence[tuple[str, Any]]) -> None:
    if rich_enabled():
        table = Table(title=title, border_style="bright_blue", header_style="bold cyan", show_header=False)
        table.add_column("Key", style="bold white")
        table.add_column("Value", style="white")
        for key, value in items:
            table.add_row(key, str(value))
        console.print(table)
        return
    print(json.dumps(dict(items), indent=2, default=str))


def run_staged_progress(title: str, steps: Sequence[tuple[str, Callable[[], T]]]) -> T:
    result: T | None = None
    if not rich_enabled():
        for _, action in steps:
            result = action()
        return result  # type: ignore[return-value]

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(title, total=len(steps))
        for description, action in steps:
            progress.update(task, description=description)
            result = action()
            progress.advance(task)
    return result  # type: ignore[return-value]
