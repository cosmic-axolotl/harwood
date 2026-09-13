"""The interactive session -- what `cl>` was, without the command language.

Section 4: importing astropy, scipy and matplotlib can cost more than a second,
and paying that on every command is the kind of friction that makes people
write one long script instead of exploring. IRAF solved the same problem by
keeping executables loaded (`prcache`); here the process simply stays alive.

The same tasks run from bash, identically. The session is a convenience, never
a second way of doing things -- and there is no scripting language to learn,
because anything a script needs is Python.
"""

from __future__ import annotations

import shlex

from rich.console import Console

PROMPT = "hw> "

console = Console()


def start() -> None:
    """Run the read-eval-print loop."""
    import click

    from .main import app

    console.print("[bold]harwood[/bold] -- interactive session. "
                  "[dim]`list` to browse, `help <task>` for the manual, "
                  "`exit` to leave.[/dim]")
    _warm()

    while True:
        try:
            line = console.input(f"[bold cyan]{PROMPT}[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue
        if line in ("exit", "quit", "bye", "logout"):
            return
        if line.startswith("!"):
            import subprocess

            subprocess.run(line[1:], shell=True)
            continue

        try:
            args = shlex.split(line)
        except ValueError as error:
            console.print(f"[red]{error}[/red] -- unbalanced quotes?")
            continue

        try:
            app.main(args=args, prog_name="harwood", standalone_mode=False)
        except click.ClickException as error:
            error.show()
        except click.exceptions.Exit:
            pass
        except SystemExit:
            pass
        except Exception as error:  # a task failing must not end the session
            console.print(f"[red]{type(error).__name__}:[/red] {error}")


def _warm() -> None:
    """Load the task registry now, so the first command is not the slow one."""
    from ..core.registry import list_tasks

    count = len(list_tasks())
    console.print(f"[dim]{count} tasks loaded.[/dim]")
