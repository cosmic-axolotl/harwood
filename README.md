# harwood

**H**omogeneous **A**stronomical **R**eduction **W**orkflows: **O**pen, **O**bservable, **D**ocumented

A modern replacement for IRAF: the same tasks, the same names, the same muscle
memory -- without the 1981 machine underneath.

## The name

**Margaret Harwood** (1885-1979) was the first woman to direct an independent
observatory, running the Maria Mitchell Observatory from 1916 to 1957. She
specialised in photographic photometry of variable stars and asteroids; her work
on Eros helped characterise that object's light variation, measured plate by
plate, by hand.

She spent forty years doing exactly what this package automates: measuring
fluxes carefully and recording how each measurement was obtained.

## Why it exists

IRAF has no equivalent. Forty-five years on, it is still the only environment
that treats a reduction as a single coherent system rather than a pile of
scripts -- tasks with persistent parameters, an interactive display wired to the
analysis, an auditable trail in every header.

It is also complex, dated, and steep. The algorithms have largely been ported to
Python (`ccdproc`, `photutils`, `specreduce`, `astroscrappy`). What has not been
ported is the environment around them.

harwood is that environment, rebuilt: Python where Python reaches, compiled
kernels where it does not, and nothing that requires you to learn a 1981 command
language first.

## Status

Early. See `docs/` for the design contract; the migration map in
`src/harwood/data/iraf_migration.yaml` tracks all 610 IRAF task names and what
harwood does with each.

```bash
harwood list                 # what runs today
harwood list --iraf          # the full IRAF inventory and its status
harwood help <task>          # the manual; answers for IRAF names too
harwood docs                 # the documentation, in the window or --browser
harwood docs --sample night  # the small night every example in it runs on
```

The documentation -- guides, a page for every task, examples written three
ways (terminal, Python, clicks) and tested on the sample night, and the
recipes -- is built from `src/harwood/docsite/` and the tasks themselves.
The window (Help, F1) and a browser show the same files.

## Install

```bash
uv tool install harwood        # or: pipx install harwood
```

No compiler, no system configuration. Extras: `harwood[ccd]`, `harwood[phot]`,
`harwood[gui]`, `harwood[docs]` (the documentation inside the window),
`harwood[fast]`.

To work on harwood itself -- on Linux, or on Windows through WSL -- see
[docs/development.md](docs/development.md).

## License

BSD 3-Clause, like the rest of the Astropy ecosystem.
