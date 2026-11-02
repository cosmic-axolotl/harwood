"""The harwood command line.

Four commands with four distinct jobs, as section 3.2 requires -- the mistake
to avoid is letting them blur into one another::

    harwood list             browse packages and tasks
    harwood <task> --help    quick reference: parameters, types, allowed values
    harwood help <task>      the manual: what it does, how, examples
    harwood lpar <task>      the values that will actually be used, and why

`--help` shows defaults; `lpar` shows *effective values and their origin*. A
`--help` that prints the default as though it were the value in use is exactly
the confusion IRAF's `lpar` existed to remove.

Nothing in this module knows about any specific task. Commands are built from
the parameter models, and built only when asked for, so that `list` and
`--help` stay instant even once there are hundreds of tasks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import build

console = Console()
error_console = Console(stderr=True)

#: Where `cd -` goes.
_previous_directory: str | None = None

#: (project, directory) already mentioned in this process.
_inherited_told: set[tuple[Path, Path]] = set()

BUILTINS = ("list", "help", "phelp", "references", "lpar",
            "init", "doctor", "log", "history", "edit", "docs",
            "gui", "cd", "pwd")

#: What the prompts pass as click's `obj`: this process goes on after the
#: command, which is what `cd` needs to mean anything.
SESSION = "session"


class Harwood(click.Group):
    """A group whose task commands are built on demand."""

    def list_commands(self, ctx):
        from ..core.registry import list_tasks

        return sorted(set(BUILTINS) | set(list_tasks()))

    def get_command(self, ctx, name):
        builtin = super().get_command(ctx, name)
        if builtin is not None:
            return builtin
        from ..core.registry import exists

        if exists(name):
            return _task_command(name)
        from ..core import packages

        # `noao`, `digiphot`, `apphot`: typed at the cl to load a package. There
        # is nothing to load, which is an answer and not a failure, so that a
        # line copied from an IRAF exercise goes on to its next command.
        if packages.named(name) is not None:
            return _package_command(name)
        return None

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if not args or args[0].startswith("-"):
                raise
            # The answer is the whole answer: click's usage and "No such
            # command" under it only repeat it, less helpfully.
            _explain_unknown(args[0])
            raise click.exceptions.Exit(2) from None


@click.group(cls=Harwood, invoke_without_command=True,
             context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--backend", help="Force a kernel backend: reference, numba or rust.")
@click.version_option(package_name="harwood")
@click.pass_context
def app(ctx: click.Context, backend: str | None) -> None:
    """harwood -- astronomical reduction with the tasks you already know."""
    if backend:
        import os

        os.environ["HARWOOD_BACKEND"] = backend
    if ctx.invoked_subcommand is None:
        from .session import start

        start()


# --------------------------------------------------------------- browsing


def _task_names(ctx, param, incomplete: str) -> list:
    """Tab after `help`, `lpar` or `log`: the tasks, with what each does."""
    from click.shell_completion import CompletionItem

    from ..core.registry import list_tasks

    return [CompletionItem(name, help=cls.description)
            for name, cls in sorted(list_tasks().items()) if name.startswith(incomplete)]


def _package_names(ctx, param, incomplete: str) -> list:
    from click.shell_completion import CompletionItem

    from ..core.registry import packages

    return [CompletionItem(name, help=", ".join(tasks))
            for name, tasks in sorted(packages().items()) if name.startswith(incomplete)]


@app.command("list")
@click.argument("package", required=False, shell_complete=_package_names)
@click.option("--iraf", is_flag=True,
              help="Show the IRAF inventory and the status of each task.")
@click.option("--milestone", "-m", help="With --iraf: filter by milestone, mvp or later.")
def list_cmd(package: str | None, iraf: bool, milestone: str | None) -> None:
    """Browse packages and tasks."""
    if iraf or milestone:
        _list_iraf(package, milestone)
        return

    from ..core.registry import list_tasks

    tasks = {n: c for n, c in list_tasks().items()
             if package is None or c.package == package}
    if not tasks:
        from ..core import packages as tree
        from ..core.registry import packages

        # An IRAF package harwood does not keep as one of its own: `noao`, or
        # `digiphot`, which holds apphot and photcal.
        node = tree.named(package)
        if node is None:
            error_console.print(f"[red]No package '{package}'.[/red] "
                                f"Known: {', '.join(sorted(packages()))}")
            raise SystemExit(2)
        tasks = {n: c for n, c in list_tasks().items() if n in tree.names_under(node)}
        if not tasks:
            _print_advice([_package_note(package)])
            return

    table = Table(title=f"harwood tasks{f' in {package}' if package else ''}",
                  title_style="bold")
    table.add_column("package", style="cyan")
    table.add_column("task", style="bold")
    table.add_column("description")
    for name, cls in tasks.items():
        label = name
        if cls.defaults:
            label = f"{name} [dim](preset)[/dim]"
        table.add_row(cls.package, label, cls.description)
    console.print(table)
    console.print("[dim]harwood list --iraf shows every IRAF task and what harwood "
                  "does with it.[/dim]")


def _list_iraf(domain: str | None, milestone: str | None) -> None:
    from ..core import migration

    style = {"available": "green", "partial": "yellow", "planned": "cyan",
             "replaced": "magenta", "wontport": "dim"}
    rows = [(n, e) for n, e in sorted(migration.tasks().items())
            if (domain is None or e["domain"] == domain)
            and (milestone is None or e.get("milestone") == milestone)]
    if not rows:
        error_console.print(f"[yellow]Nothing matches.[/yellow] Domains: "
                            f"{', '.join(sorted(migration.domains()))}")
        raise SystemExit(1)

    table = Table(title=f"IRAF 2.18.2 -> harwood ({len(rows)} tasks)", title_style="bold")
    # The identifying columns keep their width; the note takes what is left and
    # is cut with an ellipsis. Without no_wrap on the first four, one long note
    # squeezes them all to nothing and the table becomes unreadable.
    _identity_column(table, "task", 12, style="bold")
    _identity_column(table, "package", 16, style="dim")
    _identity_column(table, "domain", 14, style="cyan")
    _identity_column(table, "status", 15)
    table.add_column("note", overflow="ellipsis", no_wrap=True, max_width=48)
    for name, entry in rows:
        status = entry["status"]
        shown = f"{status} ({entry['milestone']})" if entry.get("milestone") else status
        pkg = entry["pkg"] if isinstance(entry["pkg"], str) else entry["pkg"][0]
        table.add_row(name, pkg, entry["domain"],
                      f"[{style[status]}]{shown}[/]", entry.get("note", ""))
    console.print(table)


def _manual_sections(text: str, wanted: str) -> str:
    """The parts of a manual under the headings that have the word."""
    out, keeping = [], False
    for line in text.splitlines():
        if line.startswith("#"):
            keeping = wanted.lower() in line.lower()
        if keeping:
            out.append(line)
    return "\n".join(out)


def _help_parameter(task: str, cls, name: str) -> None:
    from rich.markup import escape

    fields = cls.Params.model_fields
    key = name.replace("-", "_")
    matches = [key] if key in fields else [f for f in fields if f.startswith(key)]
    if len(matches) != 1:
        console.print(f"[red]{task} has no parameter '{name}'.[/red]"
                      + (" It could be: " + ", ".join(matches) + "." if matches else
                         f" harwood lpar {task} lists them."))
        raise SystemExit(3)
    field = fields[matches[0]]
    default = "required" if field.is_required() else _show(
        field.get_default(call_default_factory=True))
    console.print(f"[bold]{matches[0]}[/bold] [cyan]{escape(_typename(field.annotation))}"
                  f"[/cyan] = {default}")
    console.print(escape(field.description or ""))


def _search_manuals(word: str) -> None:
    from ..core import registry
    from ..docsite.builder import PAGES

    wanted = word.lower()
    if not wanted:
        console.print("[red]--search needs what to look for.[/red]")
        raise SystemExit(2)
    found = 0
    for name, cls in sorted(registry.list_tasks().items()):
        manual = PAGES / "tasks" / f"{name}.md"
        text = manual.read_text(encoding="utf-8") if manual.exists() else (cls.__doc__ or "")
        hits = [line.strip() for line in text.splitlines() if wanted in line.lower()]
        if wanted in name or wanted in (cls.description or "").lower() or hits:
            found += 1
            console.print(f"[bold]{name}[/bold] -- {cls.description}")
            if hits:
                console.print(f"  [dim]{hits[0][:90]}[/dim]", highlight=False)
    if not found:
        console.print(f"no manual has '{word}'.")
        raise SystemExit(3)


@app.command("help")
@click.argument("task", required=False, default="", shell_complete=_task_names)
@click.option("--parameter", "-p", default="",
              help="Only this parameter, what IRAF's parameter= does.")
@click.option("--section", "-s", default="",
              help="Only the sections of the manual whose heading has this.")
@click.option("--search", is_flag=True,
              help="The tasks whose manual has the word, what IRAF's search=yes does.")
def help_cmd(task: str, parameter: str, section: str, search: bool) -> None:
    """The manual for a task: what it does, and what each parameter means."""
    from ..core.registry import exists, get_class

    if search:
        _search_manuals(task)
        return
    if not task:
        console.print("[red]help needs a task[/red], or --search and what to look for.")
        raise SystemExit(2)
    if not exists(task):
        _explain_unknown(task)
        raise SystemExit(3)

    cls = get_class(task)
    if parameter:
        _help_parameter(task, cls, parameter)
        return
    console.print(f"\n[bold]{cls.name}[/bold] -- {cls.description}")
    console.print(f"[dim]package {cls.package}"
                  + (f" · IRAF {cls.iraf}" if cls.iraf and cls.iraf != cls.name else "")
                  + "[/dim]\n")
    from rich.markup import escape

    # Manuals and descriptions are plain text that quotes IRAF syntax --
    # `[SCI,1]`, `x y [frame] key` -- and rich would take `[frame]` for a
    # style tag and print nothing in its place.
    from ..docsite.builder import PAGES

    manual = PAGES / "tasks" / f"{task}.md"
    if manual.exists():
        # The manual is the task's page of the documentation, word for word.
        from rich.markdown import Markdown

        from ..docsite.examples import for_terminal

        text = for_terminal(manual.read_text(encoding="utf-8"))
        if section:
            found = _manual_sections(text, section)
            if not found:
                names = [line.lstrip("# ") for line in text.splitlines()
                         if line.startswith("#")]
                console.print(f"[red]{task} has no section with '{section}'.[/red] "
                              + ("It has: " + ", ".join(names) + "." if names
                                 else "Its manual has no headings."))
                raise SystemExit(3)
            console.print(Markdown(found))
            return
        console.print(Markdown(text))
    elif cls.__doc__:
        console.print(escape(_dedent(cls.__doc__)))
    if cls.defaults:
        values = ", ".join(f"{k}={v}" for k, v in cls.defaults.items())
        console.print(f"\n[cyan]A preset:[/cyan] {values}\n")

    table = Table(show_header=True, header_style="dim")
    table.add_column("parameter", style="bold")
    table.add_column("type", style="cyan")
    table.add_column("default")
    table.add_column("what it does")
    for name, field in cls.Params.model_fields.items():
        required = field.is_required()
        default = "[red]required[/red]" if required else _show(field.get_default(
            call_default_factory=True))
        table.add_row(name, escape(_typename(field.annotation)), default,
                      escape(field.description or ""))
    console.print(table)
    console.print(f"[dim]Effective values and where they come from: "
                  f"harwood lpar {task}. The same manual, with examples three "
                  f"ways: harwood docs {task}[/dim]\n")


@app.command("phelp")
@click.argument("task", shell_complete=_task_names)
@click.pass_context
def phelp_cmd(ctx: click.Context, task: str) -> None:
    """The manual for a task, in the pager."""
    global console

    held = console
    with open(os.devnull, "w", encoding="utf-8") as sink:
        console = Console(record=True, file=sink, width=held.width)
        try:
            ctx.invoke(help_cmd, task=task)
            text = console.export_text()
        finally:
            console = held
    click.echo_via_pager(text)


@app.command("references")
@click.argument("topic")
def references_cmd(topic: str) -> None:
    """Which tasks are about a topic: their names and what they do are searched, and
    the tasks of IRAF not here, with what became of each."""
    from ..core import migration

    found = migration.references(topic)
    for name, line in found:
        console.print(f"{escape(name)} - {escape(line)}", highlight=False, soft_wrap=True)
    if not found:
        error_console.print(f"Nothing is about '{escape(topic)}'. A shorter word finds more; "
                            "harwood list shows every task.")
        raise SystemExit(1)


@app.command("lpar")
@click.argument("task", shell_complete=_task_names)
def lpar_cmd(task: str) -> None:
    """Show the values a task will actually use, and where each came from."""
    from ..core import params as store
    from ..core.registry import exists, get_class

    if not exists(task):
        _explain_unknown(task)
        raise SystemExit(3)

    cls = get_class(task)
    resolved = store.resolve(task, cls.Params, preset=cls.defaults,
                             preset_name=task if cls.defaults else None)

    table = Table(title=f"{task} -- effective parameters", title_style="bold")
    table.add_column("parameter", style="bold")
    table.add_column("value")
    table.add_column("origin", style="dim")
    for name in cls.Params.model_fields:
        if name not in resolved:
            table.add_row(name, "[red]required, not set[/red]", "")
            continue
        entry = resolved[name]
        value = _show(entry.value)
        if entry.origin != "default":
            value = f"[yellow]{value}[/yellow]"
        table.add_row(name, value, entry.describe())
    console.print(table)

    project = store.find_project()
    console.print(f"[dim]project file: {project or 'none in this directory or above'}[/dim]")


@app.command("lines")
@click.argument("query", required=False)
@click.option("--range", "span", nargs=2, type=float,
              help="Two wavelengths: everything catalogued between them.")
@click.option("--tolerance", default=1.0, show_default=True,
              help="How far from a wavelength to look, in Angstroms.")
@click.option("--velocity", default=0.0, show_default=True,
              help="Radial velocity in km/s, applied before the lookup.")
@click.option("--origin", default="", help="stellar, nebular, interstellar, telluric.")
def lines_cmd(query: str | None, span, tolerance: float, velocity: float,
              origin: str) -> None:
    """Look up a spectral line: by wavelength, by species, or over a range.

    `harwood lines 6707.8` says what might be there. `harwood lines "Li I"`
    lists a species. `harwood lines --range 6700 6720` shows the neighbourhood,
    which is usually the question behind the first one.
    """
    from ..core import species as sp

    if origin and origin not in sp.ORIGINS:
        error_console.print(f"[red]No origin called '{origin}'.[/red] They are "
                            f"{', '.join(sp.ORIGINS)}.")
        raise SystemExit(3)

    if span:
        found = sp.within(min(span), max(span), origin=origin or None)
        _print_lines(found, f"catalogued between {min(span):.1f} and "
                            f"{max(span):.1f} A")
        return

    if not query:
        table = Table(title="species harwood can name", title_style="bold")
        table.add_column("species", style="bold")
        table.add_column("kind")
        table.add_column("lines", justify="right")
        table.add_column("in words")
        for name in sp.species():
            entries = sp.named(name)
            table.add_row(name, entries[0].kind, str(len(entries)),
                          entries[0].common or "")
        console.print(table)
        console.print("[dim]harwood lines 6563 names what is at a wavelength; "
                      "harwood lines --range 6500 6600 shows a region.[/dim]")
        return

    try:
        wavelength = float(query)
    except ValueError:
        found = sp.named(query)
        if not found:
            error_console.print(
                f"[red]Nothing called '{query}' in the catalogue.[/red] Run "
                "[bold]harwood lines[/bold] for the species it knows.")
            raise SystemExit(3) from None
        _print_lines(found, f"{query}")
        return

    match = sp.at(wavelength, tolerance, velocity, origin or None)
    if not match.candidates:
        console.print(f"Nothing catalogued within {tolerance} A of "
                      f"{wavelength:.2f} A.")
        console.print("[dim]The catalogue holds the lines that name "
                      "themselves, not every line that exists.[/dim]")
        return

    console.print(f"[bold]{wavelength:.2f} A[/bold]"
                  + (f" at {velocity:+.0f} km/s" if velocity else ""))
    for index, candidate in enumerate(match.candidates):
        line = candidate.line
        mark = "[green]>[/green]" if index == 0 else " "
        console.print(f" {mark} [bold]{line.name}[/bold] -- {line.describe()}")
        console.print(f"     {candidate.offset:+.3f} A away"
                      f" ({candidate.velocity:+.0f} km/s if this is it)")
        for text in (line.note, line.about):
            if text:
                console.print(f"     [dim]{escape(text)}[/dim]")
    if not match.confident:
        console.print("[yellow]Ambiguous:[/yellow] more than one species fits. "
                      "The wavelength alone cannot choose -- what you pointed "
                      "at can.")


def _print_lines(found, title: str) -> None:
    """A set of catalogue entries as a table."""
    if not found:
        console.print(f"Nothing {title}.")
        return
    table = Table(title=title, title_style="bold")
    table.add_column("wavelength", justify="right")
    table.add_column("species", style="bold")
    table.add_column("line")
    table.add_column("origin")
    table.add_column("note")
    for line in found:
        table.add_row(f"{line.wavelength:.2f}", line.species,
                      line.label or "", "/".join(line.origin),
                      line.note or line.about)
    console.print(table)


@app.command("standards")
@click.argument("caldir", required=False)
@click.option("--star", default="", help="Find one star, wherever it lives.")
def standards_cmd(caldir: str | None, star: str) -> None:
    """List the flux standards available, or find one.

    The same star appears in several calibration directories with different
    band widths and, sometimes, different fluxes -- they were calibrated
    against different primary standards. Which one you use is a choice, so it
    is worth seeing them side by side.
    """
    from ..core import flux

    if star:
        found = flux.where_is(star)
        if not found:
            error_console.print(
                f"[red]No standard called '{star}'.[/red] Run "
                "[bold]harwood standards[/bold] to see the directories.")
            raise SystemExit(3)
        console.print(f"[bold]{star}[/bold] is in:")
        for folder, path in found:
            bands = flux.read_bands(path)
            console.print(
                f"  [bold]{folder}[/bold]  {len(bands)} bands of "
                f"{bands[0].width:.0f} A, {bands[0].wavelength:.0f} to "
                f"{bands[-1].wavelength:.0f} A   [dim]{path}[/dim]")
        console.print(f"[dim]Pick one with: standard --caldir {found[0][0]}[/dim]")
        return

    if caldir:
        names = flux.stars(caldir)
        if not names:
            error_console.print(
                f"[red]No calibration directory called '{caldir}'[/red], or it "
                "is empty. Run [bold]harwood standards[/bold] for the list.")
            raise SystemExit(3)
        console.print(f"[bold]{caldir}[/bold] -- "
                      f"{flux.CALDIRS.get(caldir, 'a calibration directory')}")
        console.print("  " + "  ".join(name for name in names if name != "names"))
        return

    roots = [root for root in flux.IRAF_STANDARDS if root.is_dir()]
    if not roots:
        error_console.print(
            "[red]No calibration data found[/red], not even harwood's own copy of "
            "IRAF's onedstds (data/iraf/onedstds): the installation is incomplete. "
            "Reinstall harwood.")
        raise SystemExit(3)

    table = Table(title="flux standards", title_style="bold")
    table.add_column("caldir", style="bold")
    table.add_column("stars", justify="right")
    table.add_column("what it is")
    for name, what in flux.CALDIRS.items():
        found = flux.stars(name)
        if found:
            table.add_row(name, str(len([s for s in found if s != "names"])), what)
    console.print(table)
    console.print("[dim]harwood standards spec50cal lists its stars; "
                  "harwood standards --star feige110 finds one.[/dim]")


# ------------------------------------------------------------ the project


@app.command("init")
@click.argument("directory", type=click.Path(path_type=Path), default=Path("."))
@click.option("--instrument", help="Name of the instrument translation file to use.")
def init_cmd(directory: Path, instrument: str | None) -> None:
    """Start a reduction here: create a visible, commented project file."""
    from ..core import params as store

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / store.PROJECT_FILE
    if path.exists():
        error_console.print(f"[yellow]{path} already exists; leaving it alone.[/yellow]")
        raise SystemExit(1)
    store.write_project(path, instrument=instrument)
    console.print(f"[green]Created[/green] {path}")
    if store.in_home(path):
        _print_home_note()
        return
    console.print("[dim]Parameters set there apply to every task run in this "
                  "directory and below. Nothing is stored in your home directory.[/dim]")


@app.command("doctor")
def doctor_cmd() -> None:
    """Check the environment: Python, kernels, optional packages, display."""
    import os
    import platform

    from .. import kernels

    table = Table(title="harwood doctor", title_style="bold")
    table.add_column("check", style="bold")
    table.add_column("result")
    table.add_column("note", style="dim")

    table.add_row("python", platform.python_version(), sys.executable)
    for backend, version in kernels.available().items():
        table.add_row(f"kernel: {backend}", f"[green]{version}[/green]",
                      "fastest available" if backend == next(iter(kernels.available()))
                      else "")

    for missing in set(kernels.ORDER) - set(kernels.available()):
        table.add_row(f"kernel: {missing}", "[dim]not installed[/dim]",
                      escape("pip install harwood[fast]") if missing == "numba"
                      else "built from harwood-kernels; not required")
    forced = kernels.requested()
    if forced:
        table.add_row("HARWOOD_BACKEND", f"[yellow]{forced}[/yellow]",
                      "forced by the environment")

    for package, extra in (("astropy", ""), ("ccdproc", "harwood[ccd]"),
                           ("astroscrappy", "harwood[ccd]"),
                           ("scipy", "harwood[phot]"),
                           ("PySide6", "harwood[gui]"),
                           ("pyqtgraph", "harwood[gui]")):
        table.add_row(package, *_probe(package, extra))

    session = os.environ.get("WAYLAND_DISPLAY") and "wayland" or \
        (os.environ.get("DISPLAY") and "x11" or None)
    table.add_row("display", f"[green]{session}[/green]" if session
                  else "[yellow]none[/yellow]",
                  "the viewer needs one; use --backend headless over ssh"
                  if not session else "")
    console.print(table)


def _probe(package: str, extra: str) -> tuple[str, str]:
    """Whether a package is there, and how to get it if not.

    The install hint is escaped: rich reads `[phot]` as markup and eats it, so
    the one command whose job is to tell you how to fix your install was
    telling everybody to `pip install harwood`.
    """
    from importlib.metadata import PackageNotFoundError, packages_distributions, version
    from importlib.util import find_spec

    # By the module, not the distribution: harwood[gui] installs
    # PySide6-Essentials, whose module is PySide6, and asking for a
    # distribution called "PySide6" reported it missing on every install made
    # the recommended way.
    if find_spec(package) is None:
        return ("[yellow]not installed[/yellow]",
                escape(f"pip install {extra}") if extra else "")
    for distribution in [package, *packages_distributions().get(package, [])]:
        try:
            return f"[green]{version(distribution)}[/green]", ""
        except PackageNotFoundError:
            continue
    return "[green]installed[/green]", ""


@app.command("log")
@click.argument("task", required=False, shell_complete=_task_names)
@click.option("--workdir", "-w", type=click.Path(path_type=Path), default=Path("."))
@click.option("-n", "count", type=int, default=20, help="Show the last N entries.")
@click.option("--full", is_flag=True,
              help="Show every parameter of each run, and where each value came from.")
@click.option("--replay", type=int, metavar="N",
              help="Print the command that reproduces run N. Nothing else, so it "
                   "can be piped straight into a shell.")
def log_cmd(task: str | None, workdir: Path, count: int, full: bool,
            replay: int | None) -> None:
    """What has been done here. Name a task to see only its runs."""
    from ..core import provenance

    entries = (provenance.history_of(task, workdir) if task
               else provenance.read_log(workdir))
    if not entries:
        where = f" of '{task}'" if task else ""
        console.print(f"[yellow]No runs{where} recorded here yet.[/yellow]")
        return

    if replay is not None:
        if not 1 <= replay <= len(entries):
            error_console.print(
                f"[red]There are {len(entries)} runs{f' of {task}' if task else ''}; "
                f"asked for {replay}.[/red]")
            raise SystemExit(2)
        entry = entries[replay - 1]
        if not entry.get("command"):
            error_console.print("[yellow]That run predates command recording.[/yellow]")
            raise SystemExit(1)
        print(entry["command"])
        return

    shown = entries[-count:]
    if full:
        _print_full_log(shown, len(entries) - len(shown))
        return

    table = Table(title=f"provenance{f' · {task}' if task else ''}",
                  title_style="bold")
    _identity_column(table, "#", 3, justify="right", style="dim")
    _identity_column(table, "when", 20, style="dim")
    if not task:
        _identity_column(table, "task", 12, style="bold")
    table.add_column("outputs", overflow="ellipsis", no_wrap=True, max_width=42)
    _identity_column(table, "backend", 8, style="cyan")
    _identity_column(table, "s", 5, justify="right")
    for offset, entry in enumerate(shown, start=len(entries) - len(shown) + 1):
        row = [str(offset), entry["timestamp"]]
        if not task:
            row.append(entry["task"])
        outputs = entry["outputs"]
        summary = (", ".join(outputs) if len(outputs) <= 2
                   else f"{outputs[0]} and {len(outputs) - 1} more")
        row += [summary or "-", entry.get("backend") or "-",
                f"{entry['duration_s']:.2f}"]
        table.add_row(*row)
    console.print(table)
    hint = f" {task}" if task else " <task>"
    console.print(f"[dim]harwood log{hint} --full shows every parameter; "
                  f"--replay N prints the command that repeats run N.[/dim]")


def _print_full_log(entries: list[dict[str, Any]], offset: int) -> None:
    """Each run written out: what it did, with what, and how to do it again."""
    for index, entry in enumerate(entries, start=offset + 1):
        console.print(f"\n[bold]{index}.[/bold] [cyan]{entry['task']}[/cyan]  "
                      f"[dim]{entry['timestamp']}  {entry['duration_s']:.2f}s"
                      + (f"  ({entry['backend']})" if entry.get("backend") else "")
                      + "[/dim]")
        origins = entry.get("origins", {})
        table = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
        table.add_column("parameter", style="bold")
        table.add_column("value")
        table.add_column("origin", style="dim")
        for key, value in entry["params"].items():
            table.add_row(key, _show(value), origins.get(key, ""))
        console.print(table)
        for output in entry["outputs"]:
            console.print(f"    [cyan]->[/cyan] {output}")
        if entry.get("command"):
            console.print(f"    [dim]repeat:[/dim] {entry['command']}")


@app.command("history")
@click.argument("count", type=int, default=20, required=False)
@click.option("--find", "-f", "text", help="Only the lines that contain this.")
def history_cmd(count: int, text: str | None) -> None:
    """What was typed at hw>, here and in the window, numbered. Not the log:
    `harwood log` is what the tasks did and how to repeat it exactly."""
    from . import history

    lines = list(enumerate(history.load(), start=1))
    if text:
        lines = [(number, line) for number, line in lines if text in line]
    if not lines:
        console.print(f"[dim]Nothing typed contains '{escape(text)}'.[/dim]" if text
                      else "[dim]Nothing typed yet.[/dim]")
        return
    width = len(str(lines[-1][0]))
    for number, line in lines[-count:] if count > 0 else lines:
        console.print(f"[dim]{number:>{width}}[/dim]  {escape(line)}", highlight=False)


@app.command("edit")
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@click.option("--check", is_flag=True,
              help="Compare the files' columns and say what differs, without "
                   "opening anything: what Edit together checks before an edit.")
def edit_cmd(files: tuple[Path, ...], check: bool) -> None:
    """Open text files in tabs, to edit one or several at once."""
    if check:
        _check_columns(list(files))
        return
    from ..viewer import _qt

    window = _qt.window()
    if window is not None and hasattr(window, "edit_files"):
        window.edit_files([str(f) for f in files])
        return
    from ..viewer.files import run

    raise SystemExit(run([str(f) for f in files]))


def _check_columns(files: list[Path]) -> None:
    from ..core import columns

    if not files:
        error_console.print("[red]Name the files to compare:[/red] harwood edit --check "
                            "a.coo b.coo")
        raise SystemExit(2)
    shapes = []
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError as error:
            shapes.append(columns.Layout(str(path), unreadable=str(error.strerror or error)))
            continue
        if raw[:6] == b"SIMPLE":
            shapes.append(columns.Layout(str(path), unreadable="a FITS image, not a "
                                                               "text table"))
            continue
        shapes.append(columns.layout(raw.decode("utf-8", errors="replace"), str(path)))
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2, 0, 0))
    for heading in ("file", "format", "columns", "rows", "names"):
        table.add_column(heading)
    for shape in shapes:
        if shape.unreadable:
            table.add_row(escape(shape.name), "—", "", "", escape(shape.unreadable))
        else:
            table.add_row(escape(shape.name), shape.format, str(shape.width),
                          str(shape.rows), escape(" ".join(shape.names)))
    console.print(table)
    # What any column edit would have to be told: the edit itself is chosen in
    # the window, where the check is made again for that particular edit.
    found = columns.check(shapes, columns.Operation("keep_columns"))
    if found:
        console.print()
        _print_advice(found)
    else:
        console.print("\n[green]✔ alike[/green]: the same format, columns and names; "
                      "a column edit means the same thing in each.")


@app.command("docs")
@click.argument("topic", required=False, shell_complete=_task_names)
@click.option("--browser", is_flag=True,
              help="Open it in the web browser rather than in the window.")
@click.option("--build", "out", type=click.Path(path_type=Path),
              help="Write the whole site into this directory and stop: to publish it, "
                   "or to read it on a machine without harwood.")
@click.option("--sample", type=click.Path(path_type=Path),
              help="Write the sample night every example runs on into this directory.")
def docs_cmd(topic: str | None, browser: bool, out: Path | None,
             sample: Path | None) -> None:
    """The documentation: guides, every task, and examples written three ways
    (terminal, Python, clicks). The same pages in the window and a browser."""
    if sample is not None:
        from ..core import sample as night

        written = night.write(sample)
        console.print(f"[green]ok[/green] the sample night, {len(written)} files in "
                      f"{escape(str(sample))}:")
        console.print("  " + "  ".join(escape(p.name) for p in written), highlight=False)
        console.print(f"[dim]cd {escape(str(sample))} and follow `harwood docs "
                      "start/first-night`.[/dim]")
        return
    from .. import docsite

    if out is not None:
        try:
            docsite.build(out)
        except docsite.BuildError as error:
            error_console.print(f"[red]{escape(str(error))}[/red]")
            raise SystemExit(1) from None
        console.print(f"[green]ok[/green] the documentation is in {escape(str(out))}; "
                      f"open {escape(str(Path(out) / 'index.html'))}")
        return
    url = docsite.url_for(topic or "")
    if not url:
        error_console.print(f"[red]Nothing in the documentation is called "
                            f"'{escape(topic or '')}'.[/red] Name a task (daofind), a "
                            "package (noao.digiphot) or a page (concepts/outputs), or "
                            "search from the first page: harwood docs")
        raise SystemExit(2)
    try:
        where = docsite.ensure()
    except docsite.BuildError as error:
        error_console.print(f"[red]{escape(str(error))}[/red]")
        raise SystemExit(1) from None
    if not browser:
        try:
            from ..viewer import _qt
            from ..viewer.docs import available
        except ImportError:
            available = None
        window = _qt.window() if available else None
        if window is not None and hasattr(window, "show_docs"):
            window.show_docs(url)
            return
        if available and available():
            from ..viewer.docs import run

            raise SystemExit(run(url))
    import webbrowser

    page = where / url
    webbrowser.open(page.as_uri())
    console.print(f"[dim]opened {escape(page.as_uri())}[/dim]")


# ------------------------------------------------------- the working directory


def _directories(ctx, param, incomplete: str) -> list:
    """Tab after `cd`: cli/line.py fills in the names, directories only."""
    from click.shell_completion import CompletionItem

    return [CompletionItem(incomplete, type="dir")]


@app.command("pwd")
def pwd_cmd() -> None:
    """Print the working directory, where tasks read and write."""
    click.echo(os.getcwd())


@app.command("cd")
@click.argument("directory", required=False, shell_complete=_directories)
@click.pass_obj
def cd_cmd(session: str | None, directory: str | None) -> None:
    """Change the working directory of the session. Alone, go home; `cd -` goes
    back to the one before."""
    global _previous_directory

    from ..core.advice import Advice

    if session != SESSION:
        _print_advice([Advice(
            "harwood ends with the command, so a directory changed here would be "
            "forgotten at once",
            "cd in your shell, or start the session with harwood and cd at hw>",
            "problem")], where=error_console, label="cd")
        raise SystemExit(2)

    target = os.path.expanduser(os.path.expandvars(directory or "~"))
    if target == "-":
        if _previous_directory is None:
            _print_advice([Advice("there is no directory to go back to yet",
                                  "cd somewhere first", "problem")],
                          where=error_console, label="cd")
            raise SystemExit(1)
        target = _previous_directory
    before = os.getcwd()
    try:
        os.chdir(target)
    except OSError as error:
        reason = {FileNotFoundError: f"no directory '{target}' in {before}",
                  NotADirectoryError: f"'{target}' is a file, not a directory",
                  PermissionError: f"no permission to enter '{target}'"}
        _print_advice([Advice(reason.get(type(error), str(error)),
                              "Tab completes the directories that are here; pwd says "
                              "where you are", "problem")],
                      where=error_console, label="cd")
        raise SystemExit(1) from None
    _previous_directory = before
    click.echo(os.getcwd())
    _tell_inherited_project()


# ------------------------------------------------------------- the display


@app.command("gui")
@click.argument("image", required=False)
def gui_cmd(image: str | None) -> None:
    """Open the viewer, with an image if one is named."""
    from ..core import display as display_module
    from ..core import images

    screen = display_module.get("native")
    if image:
        data, header = images.read(image)
        screen.show(data, header, name=str(image))
    raise SystemExit(screen.run())


# ------------------------------------------------------------ task commands


def _task_command(name: str) -> click.Command:
    """Build the click command for one task from its parameter model."""
    from ..core.registry import get_class

    cls = get_class(name)

    def run(**kwargs: Any) -> None:
        _run_task(name, kwargs)

    # The task's own parameters win: a task that declares `workdir` gets its
    # own, and the harness does not add a second one. Click only warns about a
    # duplicate rather than failing, which is how this went unnoticed.
    declared = set(cls.Params.model_fields)
    harness = [
        option for option in (
            click.Option(["--plan"], is_flag=True,
                         help="Print the steps this would take, with real values, "
                              "and stop. IRAF's noproc."),
            click.Option(["--explain"], is_flag=True,
                         help="Open each step out: what it does to the data, why "
                              "that is the right thing, and what it came to. Works "
                              "with --plan to read it before running."),
            click.Option(["--editpar"], is_flag=True,
                         help="Open this task's parameters in a text editor, with "
                              "every one explained, and save what you change into "
                              "the project. Open it as often as you like."),
            click.Option(["--workdir", "-w"], type=click.Path(path_type=Path),
                         default=Path("."), help="Reduction directory for the log."),
        )
        if option.name not in declared
    ]

    command = build.TaskCommand(
        name=name,
        callback=run,
        params=[*build.parameters_for(cls.Params), *harness],
        help=(cls.description or "") + "\n\n" + _naming_help(name, cls)
             + "See `harwood help " + name + "` for the manual and `harwood lpar "
             + name + "` for effective values.",
        short_help=cls.description,
    )
    return command


def _naming_help(name: str, cls: Any) -> str:
    """The paragraph of --help that says how to name what a task writes."""
    fields = cls.Params.model_fields
    if "output" in fields and not fields["output"].is_required():
        return (f"Name the output, and an image with its extension: Zero.fits, not "
                f"Zero. Left unnamed it goes to output/hrwd_{name}_<date>/.\n\n")
    if any(field in fields for field in _OUTPUT_FIELDS):
        return "Write an image's extension: Zero.fits, not Zero.\n\n"
    return ""


def _print_home_note() -> None:
    from ..core import params as store
    from ..core.advice import Advice

    _print_advice([Advice(*store.HOME_NOTE, "note")])


def _tell_inherited_project(task: str = "<task>") -> None:
    """Say so when the parameters in use come from a directory above: nothing
    in a task's output shows it, and a run should not depend on a file its
    directory does not hold without saying. Once per directory in a session.
    On stderr, so that it never lands in what a task prints for a pipe."""
    from ..core import params as store
    from ..core.advice import Advice

    project = store.inherited_project()
    if project is None or (project, Path.cwd()) in _inherited_told:
        return
    _inherited_told.add((project, Path.cwd()))
    _print_advice([Advice(
        f"using {os.path.relpath(project)}, from a directory above this one",
        f"harwood lpar {task} shows what it sets; harwood init starts a project "
        "of this directory's own", "note")], where=error_console)


def _package_command(name: str) -> click.Command:
    """What an IRAF package name does when it is typed: says there is no need."""
    return click.Command(name=name, add_help_option=False,
                         context_settings={"ignore_unknown_options": True,
                                           "allow_extra_args": True},
                         callback=lambda: _print_advice([_package_note(name)]))


def _package_note(name: str) -> Any:
    from ..core import packages
    from ..core.advice import Advice

    node = packages.named(name)
    if not node.available_count():
        return Advice(f"'{name}' is an IRAF package, and none of its tasks is in "
                      "harwood yet", "harwood list --iraf says what became of each",
                      "note")
    return Advice(f"'{name}' is an IRAF package, and there is nothing to load here: "
                  "every task is always there", f"harwood list {name} shows its tasks",
                  "note")


def _run_task(name: str, kwargs: dict[str, Any]) -> None:
    from pydantic import ValidationError

    from ..core.registry import get_class

    cls = get_class(name)
    plan_only = kwargs.pop("plan", False)
    explain = kwargs.pop("explain", False)
    edit_params = kwargs.pop("editpar", False)

    # A task that declares its own `workdir` keeps it -- the value belongs to
    # the model and must not be intercepted here, or it arrives as None and the
    # log blows up after the work has already been done.
    owns_workdir = "workdir" in cls.Params.model_fields
    workdir: Path = Path(".") if owns_workdir else (
        kwargs.pop("workdir", None) or Path("."))

    if edit_params:
        _edit_parameters(name, cls)
        return

    from ..core import runner, timing

    try:
        prepared = runner.prepare(name, build.collect(cls.Params, kwargs), workdir)
    except ValidationError as error:
        _report_validation(name, error)
        raise SystemExit(2) from None
    except (ValueError, KeyError) as error:
        from rich.markup import escape

        error_console.print(f"[red]{name}:[/red] {escape(str(error))}")
        raise SystemExit(2) from None
    expected = prepared.expected
    _tell_inherited_project(name)

    if plan_only:
        console.print(f"\n[bold]{name}[/bold] would do this and nothing else:\n")
        _print_steps(prepared.plan(), explain)
        if expected:
            console.print(f"\n[dim]It should take {timing.about(expected.seconds)}, "
                          f"{expected.basis()}.[/dim]")
        console.print()
        return

    label = f"{name}... {timing.about(expected.seconds)}" if expected else f"{name}..."
    try:
        finished = runner.execute(prepared, argv=sys.argv[1:], status=console.status(label))
    except KeyboardInterrupt:
        error_console.print(f"[yellow]{name} interrupted.[/yellow]")
        raise SystemExit(130) from None
    except Exception as error:
        # A task failing is a normal event and deserves a sentence, not a
        # traceback. HARWOOD_TRACEBACK=1 brings the traceback back for
        # debugging -- hidden by default, never unavailable.
        import os

        from ..core import extras

        # The same grammar as a warning: what happened, then what to do, on
        # its own line. The suggestion may be carried by the error itself or
        # supplied by the missing-package hint. (Advice text is plain: a task's
        # message may well say "harwood[phot]", which rich would eat as markup.)
        from ..core.advice import Advice

        fix = (getattr(error, "fix", "")
               or extras.hint(error) or extras.hint(error.__cause__ or error)
               or "")
        _print_advice([Advice(str(error), fix, "problem")],
                      where=error_console, label=f"{name} stopped")
        if os.environ.get("HARWOOD_TRACEBACK"):
            console.print_exception()
        else:
            error_console.print("[dim]HARWOOD_TRACEBACK=1 shows where.[/dim]")
        raise SystemExit(1) from None
    result, folder, written = finished.result, finished.folder, finished.written

    if result.stdout is not None:
        _print_advice(result.warnings, where=error_console)
        for line in result.stdout:
            click.echo(line)
        return
    console.print(f"[green]ok[/green] {name} in {result.duration_s:.2f}s"
                  + (f" [dim]({result.backend})[/dim]" if result.backend else ""))
    if explain and result.steps:
        console.print()
        _print_steps(result.steps, True)
        console.print()
    _print_advice(result.warnings)
    for key, value in result.data.items():
        if key == "rows":
            _print_rows(value)
        elif key == "text":
            # Text a task means to be read as it is -- a grid of pixel values,
            # say -- which a table cell would wrap into nonsense.
            for line in value:
                console.print(line, markup=False, highlight=False)
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            # Records for a program to use, not a person to read: printing a
            # list of dicts is a wall of reprs. They are in the JSON log.
            console.print(f"  {key}: {len(value)} [dim](each in harwood.log.jsonl, "
                          f"and in result.data from Python)[/dim]")
        else:
            console.print(f"  {key}: {value}")
    for path in result.outputs:
        console.print(f"  [cyan]->[/cyan] {path}")
    if folder is not None:
        console.print(f"  [dim]in {folder}"
                      + (", with run.yaml" if written else "") + "[/dim]")
    _print_advice(_naming_notes(name, cls, prepared.validated, result, folder))


#: The parameters in which a task is told the name of what it writes.
_OUTPUT_FIELDS = ("output", "result", "outimage", "newnames")


def _naming_notes(name: str, cls: Any, params: Any, result: Any,
                  folder: Path | None) -> list[Any]:
    """What to know about the names of what was just written.

    At the command line only, where a person chose the name or left it out; a
    script reads `result.outputs` and has nothing to be told. Neither is a
    wrong result, so neither stops the run or goes into its record.
    """
    from ..core.advice import Advice

    fields = cls.Params.model_fields
    notes = []
    written = [Path(path) for path in result.outputs]
    if folder is not None and written and "output" in fields \
            and not getattr(params, "output", None):
        if len(written) == 1:
            notes.append(Advice(
                f"no output named, so it went to {written[0]}, and the next task will "
                "not look for it there",
                "name it, with its extension: "
                + _with_output(name, fields, written[0].name), "note"))
        else:
            notes.append(Advice(
                f"no output named, so the {len(written)} results went to {folder}, "
                "under the names of their frames, and the next task has to be told "
                "that directory",
                f"read them from {folder}/, or name the output: harwood help {name}",
                "note"))
    if any(field in fields for field in _OUTPUT_FIELDS):
        bare = [path for path in written if _is_bare_image(path)]
        if bare:
            shown = ", ".join(f"'{path}'" for path in bare[:3]) + (
                f" and {len(bare) - 3} more" if len(bare) > 3 else "")
            first = bare[0]
            notes.append(Advice(
                f"{shown}: a FITS image written as named, with no extension, so "
                f"'*.fits' and {first.name}.fits will not find it",
                f"write the extension when you name an image; for this one, "
                f"imrename {first} {first}.fits", "note"))
    return notes


def _with_output(name: str, fields: dict, value: str) -> str:
    """The task's command line with the output filled in and the rest as --help
    names them: harwood phot IMAGE COORDS obj.mag."""
    words = ["harwood", name]
    for key, field in fields.items():
        extra = field.json_schema_extra or {}
        if field.is_required() or extra.get("positional"):
            words.append(value if key == "output" else key.upper())
    if value not in words:
        words += ["--output", value]
    return " ".join(words)


def _is_bare_image(path: Path) -> bool:
    if path.suffix or not path.is_file():
        return False
    with open(path, "rb") as handle:
        return handle.read(6) == b"SIMPLE"


def _print_advice(items: list[Any], where: Any = None, label: str = "") -> None:
    """Notices, each with what happened and what to do about it.

    Colour is never the only channel: the level is spelled out in words and
    carries its own symbol, so the severity survives greyscale, a pipe into a
    file, a screen reader, and colour blindness. See `core.advice`.
    """
    import textwrap

    from rich.markup import escape

    from ..core.advice import as_advice

    where = where or console
    width = max(46, min(where.width, 100))
    for item in items:
        advice = as_advice(item)
        colour = advice.colour

        # The label occupies the first line, so it has to come out of that
        # line's budget or the text runs past the edge. textwrap is given an
        # indent of the same visible width, and the spaces are swapped for the
        # coloured label afterwards -- markup would otherwise be counted as
        # characters and every line would wrap short.
        marker = f"{advice.symbol} {label or advice.level}"
        head = "  " + marker + "  "
        body = textwrap.wrap(advice.message, width,
                             initial_indent=" " * len(head),
                             subsequent_indent="     ") or [" " * len(head)]
        where.print(f"  [{colour}]{marker}[/{colour}]  "
                    + escape(body[0][len(head):]))
        for line in body[1:]:
            where.print(escape(line))

        if advice.fix:
            lead = "     -> try: "
            for index, line in enumerate(
                    textwrap.wrap(advice.fix, width,
                                  initial_indent=" " * len(lead),
                                  subsequent_indent=" " * len(lead))):
                if index == 0:
                    where.print(f"     [{colour}]->[/{colour}] try: "
                                + escape(line[len(lead):]))
                else:
                    where.print(escape(line))


def _identity_column(table: Table, name: str, width: int, **options: Any) -> None:
    """A column that keeps its width when a neighbour holds something long.

    Rich shrinks every column to fit, so one long value in one row starves the
    columns that say *which* row it is -- and a table whose identifying columns
    have been squeezed to nothing is not a narrower table, it is an unreadable
    one. This has now been the same bug three times; hence a function.
    """
    table.add_column(name, no_wrap=True, min_width=width, **options)


def _print_steps(steps: list[Any], explain: bool) -> None:
    """One line per step; with --explain, each one opened out.

    The shape matters: the summary has to stay readable at a glance or people
    stop reading it, and the detail has to be there or the summary is just a
    prettier black box. Same content, two depths, the user picks.
    """
    import textwrap

    for index, step in enumerate(steps, start=1):
        marker = "[dim]-[/dim]" if step.skipped else "[cyan]·[/cyan]"
        body = f"[dim]{step.render()}[/dim]" if step.skipped else step.render()
        console.print(f"  {index}. {marker} {body}")
        if not explain or step.skipped:
            continue

        pad = "       "
        if step.why:
            for line in textwrap.wrap(step.why, width=76):
                console.print(f"[dim]{pad}{line}[/dim]")
        else:
            console.print(f"[dim]{pad}(no explanation written for this step yet)[/dim]")

        if step.measured:
            pairs = "   ".join(f"{key} [bold]{_show(value)}[/bold]"
                               for key, value in step.measured.items())
            console.print(f"{pad}{pairs}")
        if step.before and step.after:
            console.print(
                f"{pad}before  median [bold]{step.before.get('median', 0):.4g}[/bold]"
                f"  sigma [bold]{step.before.get('sigma', 0):.4g}[/bold]"
                f"{chr(10)}{pad}after   median [bold]{step.after.get('median', 0):.4g}[/bold]"
                f"  sigma [bold]{step.after.get('sigma', 0):.4g}[/bold]")
        console.print()


def _print_rows(rows: Any) -> None:
    """Print a task's `rows` as a table -- when that is what they are.

    `rows` is the CLI's convention for "a table to print", and a task that
    means something else by the word should not crash the command. It is a
    shared namespace and the cost of getting it wrong should be a plain line,
    not a traceback.
    """
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        console.print(f"  rows: {_show(rows)}")
        return
    if not rows:
        console.print("  [yellow]no rows[/yellow]")
        return
    table = Table(show_header=True, header_style="dim")
    names = list(rows[0])
    # The first column is what says *which* row this is -- the image, the
    # spectrum, the site -- and rich will happily squeeze it to "raw/sp00…"
    # to make room for a long note. A table whose identifying column has been
    # shortened to nothing is not a narrower table, it is an unreadable one.
    # This is the fourth time; hence doing it here rather than per task.
    widest = max((len(_show(row[names[0]])) for row in rows), default=8)
    _identity_column(table, names[0], min(max(widest, 8), 34))
    for column in names[1:]:
        table.add_column(column)
    for row in rows:
        table.add_row(*[_show(value) for value in row.values()])
    console.print(table)


def _edit_parameters(name: str, cls: Any) -> None:
    """Open the task's parameters in an editor and save what changed.

    The file is generated from the project each time, so it always shows the
    current state rather than a snapshot, and closing it without saving leaves
    everything exactly as it was.
    """
    import os
    import shutil
    import tempfile

    from ..core import params as store
    from ..core import parfile

    project = store.find_project()
    if project is None:
        project = Path.cwd() / store.PROJECT_FILE
        console.print(f"[dim]No project file here yet; saving will create "
                      f"{project}.[/dim]")

    resolved = store.resolve(name, cls.Params, preset=cls.defaults,
                             preset_name=name if cls.defaults else None)
    text = parfile.render(name, cls.Params, resolved, cls.description, project)

    directory = Path(tempfile.mkdtemp(prefix="harwood-"))
    scratch = directory / f"{name}.par"
    scratch.write_text(text, encoding="utf-8")

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    keep = False
    try:
        if editor:
            run_editor(editor, scratch)
            saved = scratch.read_text(encoding="utf-8") != text
        else:
            saved = _native_editor(scratch, name)

        if not saved:
            console.print("[dim]Nothing changed.[/dim]")
            return

        try:
            edited = parfile.parse(scratch.read_text(encoding="utf-8"), cls.Params)
        except ValueError as error:
            error_console.print(f"[red]{name}: {error}[/red]")
            error_console.print(f"[dim]Your edits are still in {scratch}.[/dim]")
            keep = True
            raise SystemExit(2) from None

        changed = parfile.apply(name, edited, cls.Params, project)
    finally:
        # The scratch copy goes, unless the message above has just told the
        # user their edits are in it.
        if not keep:
            shutil.rmtree(directory, ignore_errors=True)

    if not changed:
        console.print("[dim]Nothing changed.[/dim]")
        return
    console.print(f"[green]Saved to[/green] {project}")
    if store.in_home(project):
        _print_home_note()
    for field, (before, after) in sorted(changed.items()):
        console.print(f"  [bold]{field}[/bold]  {_show(before)} "
                      f"[dim]->[/dim] {_show(after)}")
    console.print(f"[dim]harwood lpar {name} shows the effective values.[/dim]")


def run_editor(editor: str, path: Path, windows: bool | None = None) -> None:
    """EDITOR on a file. Windows has its own rules for a command line, quotes and the
    backslashes of its paths included, which a POSIX split would take for escapes."""
    import shlex
    import subprocess

    if os.name == "nt" if windows is None else windows:
        subprocess.run(f'{editor} "{path}"', check=False)
    else:
        subprocess.run([*shlex.split(editor), str(path)], check=False)


def _native_editor(path: Path, task: str) -> bool:
    """harwood's own editor, when no EDITOR is set. A terminal fallback after that."""
    try:
        from ..viewer.editor import edit

        return edit(path, f"harwood · {task} parameters")
    except Exception:
        import shutil
        import subprocess

        for candidate in ("nano", "vim", "vi", "notepad"):
            if shutil.which(candidate):
                before = path.read_text(encoding="utf-8")
                subprocess.run([candidate, str(path)], check=False)
                return path.read_text(encoding="utf-8") != before
        error_console.print(
            "[red]No editor available.[/red] Set EDITOR, or install PySide6 "
            "for harwood's own: pip install harwood[gui]")
        raise SystemExit(1) from None


# ----------------------------------------------------------------- helpers


def _explain_unknown(name: str) -> None:
    """Answer for a name harwood does not run.

    Section 3.2: someone coming from IRAF types the name they know, and
    "command not found" does not say whether the task changed name, is coming
    later, or is never coming. The migration map answers all three -- and when
    the name is in neither place it is usually a typo, so say the nearest
    thing rather than nothing at all. Silence is the worst of the three.
    """
    if not name:
        return
    import difflib

    from ..core import migration, packages
    from ..core.registry import list_tasks

    if packages.named(name) is not None:
        _print_advice([_package_note(name)], where=error_console)
        return

    explanation = migration.explain(name)
    if explanation:
        error_console.print(f"[yellow]{explanation}[/yellow]")
        error_console.print("[dim]The whole inventory: harwood list --iraf[/dim]")
        return

    running = difflib.get_close_matches(name, list_tasks(), n=3)
    if running:
        error_console.print(f"[red]No task called '{name}'.[/red] "
                            f"Did you mean: {', '.join(running)}?")
        return

    known = difflib.get_close_matches(name, migration.tasks(), n=3)
    if known:
        error_console.print(f"[red]No task called '{name}'.[/red] "
                            f"IRAF has {', '.join(known)} -- "
                            f"try `harwood help {known[0]}`.")
        return
    error_console.print(f"[red]No task called '{name}',[/red] here or in IRAF. "
                        "Browse with `harwood list` or `harwood list --iraf`.")


def _report_validation(name: str, error: Any) -> None:
    error_console.print(f"[red]{name}: these parameters do not work.[/red]")
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "(general)"
        error_console.print(f"  [bold]{where}[/bold]: {item['msg']}")
    error_console.print(f"\n[dim]harwood help {name} lists every parameter.[/dim]")


def _typename(annotation: Any) -> str:
    import types
    import typing
    from typing import Literal, get_args, get_origin

    origin = get_origin(annotation)
    if origin is Literal:
        return "|".join(str(a) for a in get_args(annotation))
    if origin in (typing.Union, types.UnionType):
        return " or ".join(_typename(a) for a in get_args(annotation)
                           if a is not type(None))
    if origin is list:
        return f"{_typename(get_args(annotation)[0])} list"
    return getattr(annotation, "__name__", str(annotation))


def _show(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 1e5 and value == value and abs(value) != float("inf"):
            # Six significant figures turn a Julian date into 2.4613e+06 -- a
            # whole year in the last digit -- and a flux of 961592.4 into
            # 961592. Large numbers are printed whole.
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(_show(v) for v in value)
    return str(value)


def _dedent(text: str) -> str:
    import textwrap

    first, _, rest = text.partition("\n")
    return (first + "\n" + textwrap.dedent(rest)).strip()


def utf8_streams() -> None:
    """The terminal's text in UTF-8, whatever the system's code page. On Windows a redirected
    output is cp1252, which has none of the symbols a table or a warning draws, and the task
    stopped half way through its answer."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").replace("-", "").lower()
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    utf8_streams()
    app()


if __name__ == "__main__":
    main()
