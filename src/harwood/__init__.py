"""harwood -- astronomical reduction with the tasks you already know.

From Python, a task runs exactly as the command line runs it: the project
file and the preset underneath, a directory of its own under output/, an
entry in the log::

    import harwood

    result = harwood.run("daofind", image="obj001.fits", fwhmpsf=3.0)
    result.outputs          # what it wrote
    result.data             # what it measured
    result.warnings         # what to look at, each with what to do

    harwood.plan("ccdproc", input="@obj.lis", zero="Zero.fits")   # --plan

The parameter names are the task's (`harwood help daofind`); a single name
where the task takes a list is a list of one, as on the command line.
"""

from __future__ import annotations

from typing import Any


def run(task: str, /, **params: Any) -> Any:
    """Run one task. Same values, outputs and log as `harwood <task> ...`."""
    from .core.runner import run as _run

    return _run(task, **params)


def plan(task: str, /, **params: Any) -> list:
    """The steps a run would take, with real values, and nothing done: `--plan`."""
    from .core.runner import plan as _plan

    return _plan(task, **params)


__all__ = ["plan", "run"]
