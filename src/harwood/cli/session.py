"""The interactive session -- what `cl>` was, without the command language.

Section 4: importing astropy, scipy and matplotlib can cost more than a second,
and paying that on every command is the kind of friction that makes people
write one long script instead of exploring. IRAF solved the same problem by
keeping executables loaded (`prcache`); here the process simply stays alive.

The same tasks run from bash, identically. The session is a convenience, never
a second way of doing things -- and there is no scripting language to learn,
because anything a script needs is Python. What it does have is what a prompt
is expected to have: several commands on a line (`;`, `&&`, `||`, see
cli/line.py), Tab to complete, and a history that outlives the session and is
shared with the window's prompt.
"""

from __future__ import annotations

import re
import shlex
import sys

from rich.console import Console

from . import history
from . import line as command_line

PROMPT = "hw> "

console = Console()

#: A `!cd` with nothing else for its shell to do.
_BANG_CD = re.compile(r"!\s*cd(\s+[^;&|]+)?")


def start() -> None:
    """Run the read-eval-print loop."""
    console.print("[bold]harwood[/bold] -- interactive session. "
                  "[dim]`list` to browse, `help <task>` for the manual, "
                  "`exit` to leave. Tab completes; `;` and `&&` put several "
                  "commands on a line; `cd` and `pwd` as in a shell. Name your "
                  "outputs, images with their extension: Zero.fits.[/dim]")
    ask = _readline() or (lambda: console.input(f"[bold cyan]{PROMPT}[/bold cyan]"))
    _warm()

    while True:
        try:
            line = ask().strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue
        history.add(line)
        if line in ("exit", "quit", "bye", "logout"):
            return
        if _BANG_CD.fullmatch(line):
            # It would change the directory of a shell that ends at once.
            console.print("[dim]`!cd` would change only a shell of its own; doing "
                          "`cd` here.[/dim]")
            line = line[1:].strip()
        if line.startswith("!"):
            # The rest of the line is the shell's, `;` and all, as in the cl.
            import subprocess

            subprocess.run(line[1:], shell=True)
            continue
        try:
            command_line.run(line, _run, lambda note: console.print(f"[dim]{note}[/dim]"))
        except ValueError as error:
            fix = getattr(error, "fix", "")
            console.print(f"[red]{error}[/red]" + (f" -- {fix}" if fix else ""))
        except _Leave:
            return


class _Leave(Exception):
    """`exit` in the middle of a line: `imcombine ... ; exit`."""


def _run(line: str) -> bool:
    """One command; True when it worked."""
    import click

    from .main import SESSION, app

    if line in ("exit", "quit", "bye", "logout"):
        raise _Leave
    try:
        args = shlex.split(line)
    except ValueError as error:
        console.print(f"[red]{error}[/red] -- unbalanced quotes?")
        return False
    # Every message says `harwood list`, and copying it should work here too.
    if args[:1] == ["harwood"]:
        args = args[1:]
        if not args:
            return True

    try:
        app.main(args=args, prog_name="harwood", standalone_mode=False, obj=SESSION)
    except click.ClickException as error:
        error.show()
        return False
    except click.exceptions.Exit as error:
        return not error.exit_code
    except SystemExit as error:
        return error.code in (None, 0)
    except Exception as error:  # a task failing must not end the session
        console.print(f"[red]{type(error).__name__}:[/red] {error}")
        return False
    return True


def _readline():
    """Tab completion and the arrow keys, from GNU readline, at a real
    terminal; None where there is none (a pipe, a test), and the plain
    prompt is used instead.

    The prompt goes to input() itself, with the colour wrapped in \\001/\\002,
    rather than printed beforehand by rich: readline has to know how wide
    the prompt is, or it writes over it when it redraws a recalled line.
    """
    if not sys.stdin.isatty():
        return None
    try:
        import readline
    except ImportError:
        return None

    for line in history.load()[-history.LIMIT:]:
        readline.add_history(line)
    readline.set_completer_delims(" \t\n;&|")
    candidates: list[str] = []

    def completer(text: str, state: int) -> str | None:
        if state == 0:
            typed = readline.get_line_buffer()[:readline.get_endidx()]
            found = command_line.complete(typed)
            # readline replaces only the word after its own delimiters; give it
            # whole words, starting where readline thinks this one starts.
            cut = readline.get_begidx() - found.start
            candidates[:] = [item.value[cut:] if cut >= 0 else item.value
                             for item in found.items]
            if len(candidates) == 1 and not candidates[0].endswith("/"):
                candidates[0] += " "
        return candidates[state] if state < len(candidates) else None

    readline.set_completer(completer)
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    prompt = "\001\033[1;36m\002" + PROMPT + "\001\033[0m\002"
    return lambda: input(prompt)


def _warm() -> None:
    """Load the task registry now, so the first command is not the slow one."""
    from ..core.registry import list_tasks

    count = len(list_tasks())
    console.print(f"[dim]{count} tasks loaded.[/dim]")
