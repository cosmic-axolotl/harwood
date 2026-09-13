# The design contract

`design-decisions.pdf` is the contract. When an implementation question comes
up, the answer should be in there; when it is not, the answer gets added there
before it gets written in code.

`iraf-inventory.pdf` is what it derives from: the 609 task names of
Community IRAF 2.18.2, the 22 domains, what makes IRAF a reference and where it
trips.

The machine-readable half of the inventory lives in
`../src/harwood/data/iraf_migration.yaml` and is read by `harwood list --iraf`.

## Where each decision is implemented

| Section | Decision | Code |
|---|---|---|
| 2 | No hidden state; parameters in the reduction directory | `core/params.py` |
| 2.1 | Execution plan before running | `Task.plan`, `--plan` |
| 2.2 | 1-based (x, y) on every visible surface | `core/pixels.py` |
| 3 | IRAF names and meanings kept | `tasks/*.py` |
| 3.1 | POSIX command line | `cli/build.py` |
| 3.2 | list / --help / help / lpar, and the migration map | `cli/main.py`, `core/migration.py` |
| 4 | One-command install, `init`, `doctor`, session | `cli/main.py`, `cli/session.py` |
| 5 | A parameter editor that explains | `viewer/window.py` |
| 6 | English by default | everywhere |
| 7 | Numba first, Rust for stable kernels | `kernels/` |
| 8 | Native viewer, DS9 as a last resort | `core/display.py`, `viewer/` |
