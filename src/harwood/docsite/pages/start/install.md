# Install

harwood needs Python 3.11 or later. Everything it needs installs with pip;
nothing has to be configured afterwards.

```console
$ pip install "harwood[all]"
```

The extras, if you would rather pick:

| extra | what it adds |
|---|---|
| `gui` | the window: image display, the Tasks panel, the editor |
| `docs` | this documentation inside the window (QtWebEngine); without it, the pages open in your browser |
| `phot` | scipy, for imexamine's profile fits and the cosmic-ray filters |
| `ccd` | ccdproc and astroscrappy, the community packages harwood's CCD reduction and cosmic-ray cleaning are compared with; harwood runs without them |
| `fast` | numba, the compiled kernels |
| `all` | all of the above except `docs` |

## Check it

`harwood doctor` lists what is installed, what is missing, and the command
that installs it.

```example
title: Is everything there?
run: harwood doctor
```

## Open the window

```console
$ harwood gui
```

or `harwood gui obj001.fits` to start with an image on display. Without the
`gui` extra, every task still runs from the terminal and from Python.

## Without the window: the session

`harwood` on its own starts a session: the `hw>` prompt, with the tasks
loaded once so that each command starts at once. Tab completes, the arrows
recall what was typed (here and in the window), and `;` and `&&` put
several commands on a line. `exit` leaves it. See
[the command line](page:app/command-line).
