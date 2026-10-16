"""One source, two speeds: numba when it is installed, plain Python when not.

Kernels written as plain numerical Python -- loops over arrays -- are
compiled by numba.njit when numba is there (`harwood[fast]`) and
HARWOOD_BACKEND is not "reference", and run as they are otherwise. The
numbers are the same either way; only the time differs.
"""

from __future__ import annotations

import os


def available() -> bool:
    if os.environ.get("HARWOOD_BACKEND", "").strip().lower() == "reference":
        return False
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def jit(function):
    """numba.njit(cache=True) when available; the function itself otherwise."""
    if not available():
        return function
    import numba

    return numba.njit(cache=True)(function)


def backend() -> str:
    if not available():
        return "reference"
    import numba

    return f"numba {numba.__version__}"
