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
@click.option("--iraf", is_flag=True, help="Show the IRAF inventory and the status of each task.")
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
    table.add_column("task", style="bold")
    table.add_column("package", style="dim")
    table.add_column("domain", style="cyan")
    table.add_column("status")
    table.add_column("note", overflow="ellipsis", no_wrap=True)
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
    for name, field in cls.Params.model_fields.items():
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
    for missing in set(kernels.ORDER) - set(kernels.available()):
        table.add_row(f"kernel: {missing}", "[dim]not installed[/dim]",
                      "pip install harwood[fast]" if missing == "numba" else "optional")
    forced = kernels.requested()
    if forced:
        table.add_row("HARWOOD_BACKEND", f"[yellow]{forced}[/yellow]", "forced by the environment")

    for package, extra in (("astropy", ""), ("photutils", "harwood[phot]"),
                           ("ccdproc", "harwood[ccd]"), ("astroscrappy", "harwood[ccd]"),
                           ("PySide6", "harwood[gui]"), ("pyqtgraph", "harwood[gui]")):
        table.add_row(package, *_probe(package, extra))

    session = os.environ.get("WAYLAND_DISPLAY") and "wayland" or \
        (os.environ.get("DISPLAY") and "x11" or None)
    table.add_row("display", f"[green]{session}[/green]" if session
                  else "[yellow]none[/yellow]",
                  "the viewer needs one; use --backend headless over ssh"
                  if not session else "")
    console.print(table)


def _probe(package: str, extra: str) -> tuple[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"[green]{version(package)}[/green]", ""
    except PackageNotFoundError:
        return "[yellow]not installed[/yellow]", f"pip install {extra}" if extra else ""


@app.command("log")
@click.option("--workdir", "-w", type=click.Path(path_type=Path), default=Path("."))
@click.option("-n", "count", type=int, default=20, help="Show the last N entries.")
def log_cmd(workdir: Path, count: int) -> None:
    """What has been done in this reduction directory."""
    from ..core import provenance

    entries = provenance.read_log(workdir)[-count:]
    if not entries:
        console.print("[yellow]Nothing has been run here yet.[/yellow]")
        return
    table = Table(title="provenance", title_style="bold")
    table.add_column("when", style="dim")
    table.add_column("task", style="bold")
    table.add_column("outputs")
    table.add_column("backend", style="cyan")
    table.add_column("s", justify="right")
    for entry in entries:
        table.add_row(entry["timestamp"], entry["task"],
                      ", ".join(entry["outputs"]) or "-",
                      entry.get("backend") or "-",
                      f"{entry['duration_s']:.2f}")
    console.print(table)


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

    if wait and hasattr(screen, "run"):
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

    command = click.Command(
        name=name,
        callback=run,
        params=[
            *build.parameters_for(cls.Params),
            click.Option(["--plan"], is_flag=True,
                         help="Print the steps this would take, with real values, "
                              "and stop. IRAF's noproc."),
            click.Option(["--workdir", "-w"], type=click.Path(path_type=Path),
                         default=Path("."), help="Reduction directory for the log."),
        ],
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

    plan_only = kwargs.pop("plan", False)
    workdir: Path = kwargs.pop("workdir", Path("."))
    cls = get_class(name)

    try:
        given = build.collect(cls.Params, kwargs)
        resolved = store.resolve(name, cls.Params, overrides=given,
                                 preset=cls.defaults,
                                 preset_name=name if cls.defaults else None)
        validated = cls.Params(**store.values(resolved))
    except ValidationError as error:
        _report_validation(name, error)
        raise SystemExit(2) from None
    except (ValueError, KeyError) as error:
        error_console.print(f"[red]{name}:[/red] {error}")
        raise SystemExit(2) from None

    task = get_task(name)

    if plan_only:
        console.print(f"\n[bold]{name}[/bold] would do this and nothing else:\n")
        for index, step in enumerate(task.plan(validated), start=1):
            marker = "[dim]-[/dim]" if step.skipped else "[cyan]·[/cyan]"
            body = f"[dim]{step.render()}[/dim]" if step.skipped else step.render()
            console.print(f"  {index}. {marker} {body}")
        console.print()
        return

    try:
        with console.status(f"{name}..."):
            result = task.run(validated)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        error_console.print(f"[red]{name} stopped:[/red] {error}")
        raise SystemExit(1) from None

    provenance.record(name, validated.model_dump(), result, workdir,
                      origins=store.origins(resolved))

    console.print(f"[green]ok[/green] {name} in {result.duration_s:.2f}s"
                  + (f" [dim]({result.backend})[/dim]" if result.backend else ""))
    for key, value in result.data.items():
        if key == "rows":
            _print_rows(value)
        else:
            console.print(f"  {key}: {value}")
    for output in result.outputs:
        console.print(f"  [cyan]->[/cyan] {output}")


def _print_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        console.print("  [yellow]no rows[/yellow]")
        return
    table = Table(show_header=True, header_style="dim")
    for column in rows[0]:
        table.add_column(column)
    for row in rows:
        table.add_row(*[_show(value) for value in row.values()])
    console.print(table)


# ----------------------------------------------------------------- helpers


def _explain_unknown(name: str) -> None:
    """Answer for a name harwood does not run.

    Section 3.2: someone coming from IRAF types the name they know, and
    "command not found" does not say whether the task changed name, is coming
    later, or is never coming. The migration map answers all three.
    """
    if not name:
        return
    from ..core import migration

    explanation = migration.explain(name)
    if explanation:
        error_console.print(f"[yellow]{explanation}[/yellow]")
        error_console.print("[dim]The whole inventory: harwood list --iraf[/dim]")


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
