"""Examples: written once, as the command line, and shown three ways.

A page writes an example as a fenced block::

    ```example
    title: Find the stars on one frame
    after: reduced
    run: harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3 --threshold 5
    output: |
      sources: 30
    ```

`run` is one command line or a list of them. The Python tab and the Clicks
tab are not written at all: they are worked out from the command line --
through the task's own click command, so the values are the ones the
command line would pass -- and so the three cannot drift apart. `after`
names the state of the sample night the example starts from (`night`, the
raw frames; `reduced`, processed; `found`, with obj001.coo), so that it can
be followed by hand and run by the tests. `output` is a few lines the run
prints, shown under the command; the tests check that it does print them,
so a number on a page is a number harwood produced. `fails: true` marks an
example that shows a failure on purpose -- what harwood says when a file is
missing -- and the tests then check that it does fail.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..core.i18n import N_

#: Options every task has that are not parameters of the task.
HARNESS = ("plan", "explain", "editpar", "workdir")


@dataclass
class Example:
    title: str
    lines: list[str]
    after: str = "night"
    output: str = ""
    page: str = ""                     #: the page it is on, for a failing test to name
    fails: bool = False                #: shows a failure on purpose: the last line must fail
    notes: list[str] = field(default_factory=list)


def parse(text: str, page: str = "") -> Example:
    """An example block's body. Raises ValueError saying what is wrong."""
    try:
        spec = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"{page}: an example is not valid YAML ({error})") from None
    if not isinstance(spec, dict) or "run" not in spec:
        raise ValueError(f"{page}: an example needs a 'run:' line")
    lines = spec["run"] if isinstance(spec["run"], list) else [spec["run"]]
    after = spec.get("after", "night")
    if after not in SETUPS:
        raise ValueError(f"{page}: 'after: {after}' is not one of {', '.join(SETUPS)}")
    unknown = set(spec) - {"title", "run", "after", "output", "fails"}
    if unknown:
        raise ValueError(f"{page}: an example does not take {', '.join(sorted(unknown))}")
    return Example(str(spec.get("title", "")), [str(line).strip() for line in lines],
                   after, str(spec.get("output", "")).rstrip(), page,
                   fails=bool(spec.get("fails", False)))


# --------------------------------------------------------------- the tabs


@dataclass
class Call:
    """One command line, understood."""

    line: str
    task: str                          #: the task, or the built-in command
    is_task: bool
    given: dict[str, Any] = field(default_factory=dict)   #: what was typed, by field
    plan: bool = False


def understand(line: str) -> Call:
    """Read a command line the way harwood does. Raises ValueError for one it
    would refuse -- an example that cannot run must not reach a page."""
    import click

    from ..cli import build
    from ..cli.main import BUILTINS, _task_command
    from ..core.registry import exists, get_class

    words = shlex.split(line)
    if words and words[0] == "harwood":
        words = words[1:]
    if not words:
        raise ValueError("an empty command")
    name = words[0]
    if name in BUILTINS:
        return Call(line, name, False)
    if not exists(name):
        raise ValueError(f"no task called '{name}'")
    command = _task_command(name)
    try:
        context = command.make_context(name, list(words[1:]))
    except click.ClickException as error:
        raise ValueError(f"{line}: {error.format_message()}") from None
    params = dict(context.params)
    plan = bool(params.get("plan"))
    for key in HARNESS:
        if key not in get_class(name).Params.model_fields:
            params.pop(key, None)
    return Call(line, name, True, build.collect(get_class(name).Params, params), plan)


def python(lines: list[str]) -> str | None:
    """The Python for a list of command lines, or None when one of them is a
    command for reading (list, help) with nothing to call."""
    calls = [understand(line) for line in lines]
    if not all(call.is_task for call in calls):
        return None
    out = ["import harwood", ""]
    for call in calls:
        function = "harwood.plan" if call.plan else "harwood.run"
        arguments = [repr(call.task)] + [f"{key}={_value(value)}"
                                         for key, value in call.given.items()]
        one_line = f"{function}({', '.join(arguments)})"
        if len(one_line) <= 79:
            out.append(one_line)
        else:
            out.append(f"{function}(")
            out += [f"    {argument}," for argument in arguments]
            out.append(")")
    return "\n".join(out)


def _value(value: Any) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str | Path):
        value = str(value[0])          # harwood.run takes one name for a list of one
    if isinstance(value, list):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    return repr(value)


def clicks(lines: list[str]) -> list[list[str]] | None:
    """What to press in the window, one list of steps per command line, or
    None when a line is not a task."""
    from ..core import packages
    from ..core.registry import get_class

    calls = [understand(line) for line in lines]
    if not all(call.is_task for call in calls):
        return None
    root = packages.tree()
    out = []
    for call in calls:
        where = packages.locate(call.task, root)
        path = ["all"] + (where[0].split(".") if where else [])
        # in English, marked: the documentation says them in its language (N_)
        steps = [N_(f"In Tasks, go to {' › '.join(path)} and press {call.task} "
                    f"(or type {call.task} in the search box).")]
        fields = get_class(call.task).Params.model_fields
        required = [k for k in call.given if fields[k].is_required()
                    or (fields[k].json_schema_extra or {}).get("positional")]
        optional = [k for k in call.given if k not in required]
        for key in required:
            steps.append(f"{key}: {_shown(call.given[key])}")
        if optional:
            steps.append(N_("Open More options, then " + "; ".join(
                f"{key}: {_shown(call.given[key])}" for key in optional) + "."))
        steps.append(N_("Press Plan to see the steps.") if call.plan
                     else N_("Press Run (or Plan first, to see the steps)."))
        out.append(steps)
    return out


def _shown(value: Any) -> str:
    if value is True:
        return "ticked"
    if value is False:
        return "not ticked"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


# ------------------------------------------------------------ the sample night

#: The states of the sample night an example can start from, each with the
#: commands that take the one before it there -- the reduction of
#: start/first-night, step by step. Command lines, so that the page can say
#: how to get there and the tests get there the same way.
STAGES: dict[str, tuple[str, list[str]]] = {
    "night": (N_("the sample night, as written"), []),
    "zero": (N_("the sample night with its bias combined (Zero.fits)"), [
        "harwood ccdproc @bias.lis --overscan --trim --output bias",
        "harwood zerocombine bias/bias00*.fits Zero.fits"]),
    "flat": (N_("the sample night with its bias and flat combined (Zero.fits, "
                "FlatV.fits)"), [
        "harwood ccdproc @flat.lis --overscan --trim --zerocor --zero Zero.fits "
        "--output flat",
        "harwood flatcombine flat/flat00*.fits Flat.fits"]),
    "reduced": (N_("the sample night after its reduction (Zero.fits, FlatV.fits, red/)"), [
        "harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits "
        "--flat FlatV.fits --output red"]),
    "found": (N_("the reduced night with the stars of obj001 found (obj001.coo)"), [
        "harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3"]),
}
SETUPS = tuple(STAGES)
STARTS = {name: said for name, (said, _) in STAGES.items()}


def path_to(after: str) -> list[str]:
    """Every command from the raw night to the state `after`."""
    lines: list[str] = []
    for name, (_, commands) in STAGES.items():
        lines += commands
        if name == after:
            break
    return lines


def prepare(after: str, directory: Path) -> None:
    """Bring a directory to the state an example starts from."""
    import contextlib
    import io
    import os

    import click

    from ..cli.main import app
    from ..core import sample

    sample.write(directory)
    previous = Path.cwd()
    os.chdir(directory)
    try:
        for line in path_to(after):
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    app.main(shlex.split(line)[1:], prog_name="harwood",
                             standalone_mode=False)
                except SystemExit as stop:
                    if stop.code not in (None, 0):
                        raise RuntimeError(f"getting to '{after}': {line} failed") from None
                except click.ClickException as error:
                    raise RuntimeError(f"getting to '{after}': {line}: "
                                       f"{error.format_message()}") from None
    finally:
        os.chdir(previous)


# ------------------------------------------------------------ in the terminal


def for_terminal(text: str) -> str:
    """A page's Markdown as `harwood help` prints it: an example as its command
    lines and what they print, a note as a labelled paragraph, a link as its
    words. The same text as the page, so the terminal and the documentation
    cannot say different things."""
    import re

    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^```(\w+)\s*$", line)
        if not fence:
            out.append(line)
            i += 1
            continue
        kind = fence.group(1)
        body = []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i])
            i += 1
        i += 1
        content = "\n".join(body)
        if kind == "example":
            try:
                example = parse(content)
            except ValueError:
                continue
            title = f"Example: {example.title}" if example.title else "Example"
            out += [f"**{title}**", "", "```console",
                    *[f"$ {command}" for command in example.lines]]
            out += [*example.output.splitlines(), "```"] if example.output else ["```"]
        elif kind in ("note", "tip", "warning", "problem"):
            out += [f"**{kind.title()}.** " + " ".join(content.split())]
        elif kind == "generated":
            continue
        else:
            out += [f"```{kind}", content, "```"]
    text = "\n".join(out)
    # [words](task:x) and friends: the words; the link means nothing here.
    return re.sub(r"\[([^\]]+)\]\((?:task|page|package):[^)]+\)", r"\1", text)
