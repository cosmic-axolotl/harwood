# Parameters, and where they come from

A parameter's name is IRAF's: `fwhmpsf`, `low_reject`, `epadu`. On the
command line the ones a task cannot run without are written in order, and
the rest as options:

```console
$ harwood imcombine @obj.lis final.fits --combine median --reject ccdclip
```

`low_reject` is written `--low-reject` or `--low_reject`, whichever your
fingers know; a list is written `3,5,10`, as IRAF writes it; a yes-or-no is
`--overscan` or `--no-overscan`.

## Four places, in order

When a task runs, each parameter takes the first value it finds here:

1. **the command line** (or the form, or `harwood.run(...)`);
2. **the project file**, `harwood.yaml`, in this directory or the nearest
   one above it;
3. **the preset**, for a task that is another with settings of its own --
   `zerocombine` is `imcombine` with `reject=minmax`, `nhigh=1`;
4. **the task's default**.

There is no fifth: nothing is kept in your home directory, so a night's
directory and its `harwood.yaml` are the whole account of how it was
reduced. `harwood lpar <task>` shows the value each parameter will take and
which of the four it came from.

```example
title: zerocombine's settings come from its preset
run: harwood lpar zerocombine
output: |
  preset: zerocombine
```

## The project file

`harwood init` writes a `harwood.yaml` with comments saying how it works.
It holds parameters per task:

```yaml
params:
  daofind:
    fwhmpsf: 3.0
  phot:
    epadu: 2.0
    apertures: [4, 6]
```

and from then on every `daofind` run in this directory, and below it, uses
3.0 unless told otherwise. `--editpar` edits the same file for one task,
with every parameter explained, and keeps only what you changed.

## The instrument

The CCD tasks read the image type, the filter and the sections from the
headers, through a translation file -- IRAF's `ccddb` files, such as
`ccddb$kpno/direct.dat`. [setinstrument](task:setinstrument) sets one for
the project. Without one, harwood reads `IMAGETYP`, `FILTERS`, `EXPTIME`,
`BIASSEC` and `TRIMSEC` under their own names, as the sample night writes
them.
