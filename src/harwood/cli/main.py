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

import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from . import build

console = Console()
error_console = Console(stderr=True)

BUILTINS = ("list", "help", "lpar", "init", "doctor", "recipe", "log", "display", "gui")


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
        return None

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            _explain_unknown(args[0] if args else "")
            raise


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


@app.command("list")
@click.argument("package", required=False)
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
        from ..core.registry import packages

        error_console.print(f"[red]No package '{package}'.[/red] "
                            f"Known: {', '.join(sorted(packages()))}")
        raise SystemExit(2)

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


@app.command("help")
@click.argument("task")
def help_cmd(task: str) -> None:
    """The manual for a task: what it does, and what each parameter means."""
    from ..core.registry import exists, get_class

    if not exists(task):
        _explain_unknown(task)
        raise SystemExit(3)

    cls = get_class(task)
    console.print(f"\n[bold]{cls.name}[/bold] -- {cls.description}")
    console.print(f"[dim]package {cls.package}"
                  + (f" · IRAF {cls.iraf}" if cls.iraf and cls.iraf != cls.name else "")
                  + "[/dim]\n")
    if cls.__doc__:
        console.print(_dedent(cls.__doc__))
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
        table.add_row(name, _typename(field.annotation), default, field.description or "")
    console.print(table)
    console.print(f"[dim]Effective values and where they come from: "
                  f"harwood lpar {task}[/dim]\n")


@app.command("lpar")
@click.argument("task")
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


@app.command("observatories")
@click.argument("name", required=False)
def observatories_cmd(name: str | None) -> None:
    """List the observatories setairmass knows, or look one up.

    IRAF keeps these in obsdb.dat and harwood keeps the same values built in,
    so a header saying `OBSERVAT = 'KPNO'` is enough to compute an airmass.
    """
    from ..core import observatory

    if name:
        site = observatory.find(name)
        if site is None:
            error_console.print(
                f"[red]No observatory called '{name}'.[/red] Run "
                "[bold]harwood observatories[/bold] for the list, or give "
                "setairmass --latitude and --longitude directly.")
            raise SystemExit(3)
        console.print(f"[bold]{site.code}[/bold]  {site.describe()}")
        console.print(f"[dim]longitude positive west, as IRAF stores it; "
                      f"{site.east:+.3f} counted east.[/dim]")
        return

    table = Table(title="observatories", title_style="bold")
    table.add_column("code", style="bold")
    table.add_column("name")
    table.add_column("latitude", justify="right")
    table.add_column("longitude W", justify="right")
    table.add_column("height", justify="right")
    for code in observatory.listing():
        site = observatory.SITES[code]
        table.add_row(code, site.name, f"{site.latitude:+.4f}",
                      f"{site.longitude:+.4f}", f"{site.altitude:.0f} m")
    console.print(table)
    console.print("[dim]Longitude is positive west, as IRAF stores it. For a "
                  "site not listed, setairmass takes --latitude and "
                  "--longitude.[/dim]")


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
    from rich.markup import escape

    for missing in set(kernels.ORDER) - set(kernels.available()):
        table.add_row(f"kernel: {missing}", "[dim]not installed[/dim]",
                      escape("pip install harwood[fast]") if missing == "numba"
                      else "built from harwood-kernels; not required")
    forced = kernels.requested()
    if forced:
        table.add_row("HARWOOD_BACKEND", f"[yellow]{forced}[/yellow]",
                      "forced by the environment")

    for package, extra in (("astropy", ""), ("photutils", "harwood[phot]"),
                           ("ccdproc", "harwood[ccd]"),
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
    from importlib.metadata import PackageNotFoundError, version

    from rich.markup import escape

    try:
        return f"[green]{version(package)}[/green]", ""
    except PackageNotFoundError:
        return ("[yellow]not installed[/yellow]",
                escape(f"pip install {extra}") if extra else "")


@app.command("log")
@click.argument("task", required=False)
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


# ------------------------------------------------------------- the display


@app.command("display")
@click.argument("image")
@click.option("--frame", "-f", type=int, default=1, help="Frame to load the image into.")
@click.option("--scale", type=click.Choice(["zscale", "minmax", "99.5%"]),
              default="zscale", help="How the display limits are chosen.")
@click.option("--backend", type=click.Choice(["native", "headless", "ds9"]),
              help="Where to draw. Detected from the environment when not given.")
@click.option("--coords", type=click.Path(path_type=Path),
              help="Overlay a coordinate catalogue, as tvmark does.")
@click.option("--radius", type=float, default=6.0, help="Radius of the overlay circles.")
@click.option("--wait/--no-wait", default=True,
              help="Keep the window open and interactive. --no-wait returns at once.")
def display_cmd(image: str, frame: int, scale: str, backend: str | None,
                coords: Path | None, radius: float, wait: bool) -> None:
    """Load an image into the viewer."""
    from ..core import display as display_module
    from ..core import images

    data, header = images.read(image)
    screen = display_module.get(backend)
    screen.show(data, header, frame=frame, name=str(image), scale=scale)

    if coords:
        from astropy.table import Table

        table = Table.read(str(coords))
        screen.mark([display_module.Marker(float(row["x"]), float(row["y"]),
                                          radius=radius, label=str(row.get("id", "")))
                     for row in table], frame=frame)
        console.print(f"[dim]{len(table)} positions marked from {coords}[/dim]")

    if wait and hasattr(screen, "run") and getattr(screen, "owns", True):
        raise SystemExit(screen.run())
    if hasattr(screen, "output"):
        console.print(f"  [cyan]->[/cyan] {screen.output}")


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

    command = click.Command(
        name=name,
        callback=run,
        params=[*build.parameters_for(cls.Params), *harness],
        help=(cls.description or "") + "\n\nSee `harwood help "
             + name + "` for the manual and `harwood lpar " + name
             + "` for effective values.",
        short_help=cls.description,
    )
    return command


def _run_task(name: str, kwargs: dict[str, Any]) -> None:
    from pydantic import ValidationError

    from ..core import params as store
    from ..core import provenance
    from ..core.registry import get_class, get_task

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

    try:
        given = build.collect(cls.Params, kwargs)
        resolved = store.resolve(name, cls.Params, overrides=given,
                                 preset=cls.defaults,
                                 preset_name=name if cls.defaults else None)
        validated = cls.Params(**store.values(resolved))
        if owns_workdir:
            workdir = Path(getattr(validated, "workdir", ".") or ".")
    except ValidationError as error:
        _report_validation(name, error)
        raise SystemExit(2) from None
    except (ValueError, KeyError) as error:
        from rich.markup import escape

        error_console.print(f"[red]{name}:[/red] {escape(str(error))}")
        raise SystemExit(2) from None

    task = get_task(name)

    if plan_only:
        console.print(f"\n[bold]{name}[/bold] would do this and nothing else:\n")
        _print_steps(task.plan(validated), explain)
        console.print()
        return

    from ..core import output, replay

    # try/finally, not a bare call at the end: a task that raises must not
    # leave this run's directory set for whatever runs next in the same
    # process -- the session, a recipe, a test.
    token = output.begin(name, workdir)
    try:
        try:
            with console.status(f"{name}..."):
                result = task.run(validated)
        except KeyboardInterrupt:
            error_console.print(f"[yellow]{name} interrupted.[/yellow]")
            raise SystemExit(130) from None
        except Exception as error:
            # A task failing is a normal event and deserves a sentence, not a
            # traceback. HARWOOD_TRACEBACK=1 brings the traceback back for
            # debugging -- hidden by default, never unavailable.
            import os

            from rich.markup import escape

            from ..core import extras

            # A task's message is plain text and may well contain brackets --
            # "harwood[phot]" is exactly the sort of thing it says -- and rich
            # would read those as markup and eat them.
            error_console.print(f"[red]{name} stopped:[/red] {escape(str(error))}")
            advice = extras.hint(error) or extras.hint(error.__cause__ or error)
            if advice:
                error_console.print(f"  [yellow]{escape(advice)}[/yellow]")
            if os.environ.get("HARWOOD_TRACEBACK"):
                console.print_exception()
            else:
                error_console.print("[dim]HARWOOD_TRACEBACK=1 shows where.[/dim]")
            raise SystemExit(1) from None

        command = replay.command(name, cls.Params, validated.model_dump())
        provenance.record(
            name, validated.model_dump(), result, workdir,
            origins=store.origins(resolved), command=command, argv=sys.argv[1:],
        )
        written = output.record(name, validated.model_dump(), result, command,
                                store.origins(resolved))
        folder = output.current()
    finally:
        output.end(token)

    console.print(f"[green]ok[/green] {name} in {result.duration_s:.2f}s"
                  + (f" [dim]({result.backend})[/dim]" if result.backend else ""))
    if explain and result.steps:
        console.print()
        _print_steps(result.steps, True)
        console.print()
    for warning in result.warnings:
        console.print(f"  [yellow]note:[/yellow] {warning}")
    for key, value in result.data.items():
        if key == "rows":
            _print_rows(value)
        else:
            console.print(f"  {key}: {value}")
    for path in result.outputs:
        console.print(f"  [cyan]->[/cyan] {path}")
    if folder is not None:
        console.print(f"  [dim]in {folder}"
                      + (", with run.yaml" if written else "") + "[/dim]")


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
    for column in rows[0]:
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
    import shlex
    import subprocess
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
    try:
        if editor:
            subprocess.run([*shlex.split(editor), str(scratch)], check=False)
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
            raise SystemExit(2) from None

        changed = parfile.apply(name, edited, cls.Params, project)
    finally:
        pass

    if not changed:
        console.print("[dim]Nothing changed.[/dim]")
        return
    console.print(f"[green]Saved to[/green] {project}")
    for field, (before, after) in sorted(changed.items()):
        console.print(f"  [bold]{field}[/bold]  {_show(before)} "
                      f"[dim]->[/dim] {_show(after)}")
    console.print(f"[dim]harwood lpar {name} shows the effective values.[/dim]")


def _native_editor(path: Path, task: str) -> bool:
    """harwood's own editor, when no EDITOR is set. A terminal fallback after that."""
    try:
        from ..viewer.editor import edit

        return edit(path, f"harwood · {task} parameters")
    except Exception:
        import shutil
        import subprocess

        for candidate in ("nano", "vim", "vi"):
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

    from ..core import migration
    from ..core.registry import list_tasks

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
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(_show(v) for v in value)
    return str(value)


def _dedent(text: str) -> str:
    import textwrap

    first, _, rest = text.partition("\n")
    return (first + "\n" + textwrap.dedent(rest)).strip()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
