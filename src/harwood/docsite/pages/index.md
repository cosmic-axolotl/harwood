# harwood

The tasks you know from IRAF -- `ccdproc`, `imcombine`, `daofind`, `phot`,
`apall`, `identify` -- with the same names and the same parameters, in
Python, with a window to look at the data in and nothing to configure.

Every task can be done three ways, and they are the same thing:

- **typed**, in a terminal or at the window's prompt:
  `harwood daofind obj001.fits --fwhmpsf 3`;
- **from Python**: `harwood.run("daofind", image="obj001.fits", fwhmpsf=3.0)`;
- **by clicks**, in the window's Tasks panel, where the form shows the
  command it amounts to as you fill it in.

The examples in this documentation show all three. Pick the tab you read
best; the choice follows you from page to page.

```example
title: What is in a night
run: harwood ccdlist @obj.lis
output: |
  obj001.fits
```

## Where to begin

- New to harwood: [install it](page:start/install), get
  [the sample night](page:start/sample), and [reduce it](page:start/first-night)
  from raw frames to magnitudes. It takes a few minutes.
- Coming from IRAF: [what changed and what did not](page:iraf/migration). The
  short version: the names are the same, `epar` is `--editpar`, `lpar` is
  `lpar`, and there are no packages to load.
- Looking for a task: the column on the left is IRAF's package tree,
  `noao › digiphot › apphot › daofind`, or search with `/`.

## What harwood is careful about

- **It says what it will do before it does it.** `--plan` lists the steps
  with the real values and stops; `--explain` says why each step is there.
  See [the commands](page:concepts/commands).
- **It writes down what it did.** Every run gets its own directory with a
  `run.yaml`, an entry in the log, and HISTORY in the headers it wrote. See
  [where results go](page:concepts/outputs).
- **Every warning says what to do.** And never by colour alone. See
  [notices](page:concepts/notices).
- **It is checked against IRAF.** The ported tasks are compared with IRAF
  2.18's own output, number by number, in the test suite.
