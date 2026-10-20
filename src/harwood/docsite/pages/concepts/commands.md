# The commands for reading

Four commands answer four different questions, and keeping them apart is
the point -- IRAF's `lpar` existed because a help page that shows the
default reads as though it were the value in use.

| command | answers |
|---|---|
| `harwood list` | what tasks are there? (`list apphot`: in one package; `list --iraf`: every IRAF task and what became of it) |
| `harwood <task> --help` | how do I write it: its parameters, types, defaults, allowed values |
| `harwood help <task>` | what does it do: the manual, the same text as the task's page here |
| `harwood lpar <task>` | what will it use *here*: the value of every parameter and where it came from |

```example
title: The values daofind would use, and where each comes from
run: harwood lpar daofind
output: |
  fwhmpsf
```

`lpar` says *default*, *project: harwood.yaml*, *preset: zerocombine* or
*command line* beside each value; see [parameters](page:concepts/parameters).

## Before and after a run

Every task takes three more options:

- `--plan` lists the steps it would take, with the real values -- which
  overscan strip, which bias, which images -- and stops. Nothing is
  written. It reads the headers, so a step that will be skipped is shown as
  skipped.
- `--explain` opens each step out: what it does to the data, and why it is
  the right thing to do. With `--plan`, read it before running; without,
  each step also says what it came to.
- `--editpar` opens the task's parameters in an editor, each explained, and
  saves what you change into the project -- IRAF's `epar`.

```example
title: What ccdproc would do to the bias frames, and why
run: harwood ccdproc @bias.lis --overscan --trim --plan --explain
output: |
  overscan: legendre order 1, 1 rejection passes over [105:120,1:100]
```

## After

- `harwood log` lists what was run in this directory; `log <task>` only that
  task, `log --full` every parameter and where it came from, and
  `log --replay N` the command that repeats run N exactly. See
  [where results go](page:concepts/outputs).
- `harwood history` lists what was *typed*, here and in the window.

## The rest

| command | |
|---|---|
| `harwood init` | write a commented `harwood.yaml` in this directory |
| `harwood doctor` | what is installed, and what to install |
| `harwood edit` | text files in tabs; `edit --check` compares tables ([more](page:app/files)) |
| `harwood docs` | this documentation: in the window, or `--browser`; `--sample DIR` writes the sample night |
| `harwood gui` | the window |
| `harwood` | the `hw>` session |
