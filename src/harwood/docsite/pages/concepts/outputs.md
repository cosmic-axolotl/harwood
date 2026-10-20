# Where results go, and the record of them

## A directory for each run

A task told where to write writes there. A task *not* told gets a directory
of its own, so that results never pile up in the working directory under
names that say nothing about where they came from:

```text
output/hrwd_ccdproc_2026_09_13/
output/hrwd_ccdproc_2026_09_13_2/      a second run the same day
```

The files inside keep the names of the frames they came from, because
`obj001.fits` is the only thing that says which frame it is. Beside them,
`run.yaml` says what made them: the command that repeats the run, every
parameter, and where each value came from. Hand the directory to someone
and they have the results and how they were made.

```example
title: Process the object frames into a directory of their own
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits
output: |
  with run.yaml
```

## Three records

| record | for | kept |
|---|---|---|
| `harwood.log.jsonl` | every run: parameters, their origins, the outputs, the time, the command that repeats it | in the reduction directory; `harwood log` reads it |
| `harwood.log` | a plain text line per frame per step, as IRAF's `logfile`: watch it grow in another window | in the reduction directory; `--logfile` names another, empty keeps none |
| HISTORY in the headers | what was done to *that* frame | in each image harwood writes, so it travels with the data |

```example
title: Every ccdproc run here, every value, and where it came from
after: reduced
run: harwood log ccdproc --full
output: |
  repeat:
```

## Marks in the headers

The CCD steps leave a mark in the header of each frame they process
(`OVERSCAN`, `TRIM`, `ZEROCOR`, `DARKCOR`, `FLATCOR`, `FIXPIX`), with the
time, and a step whose mark is there is not done again. Subtracting a bias
twice is the mistake this exists to prevent, as it does in IRAF.

## Never over the input

No task writes over the images it read unless told to (`ccdproc
--inplace`, as IRAF does by default). The marks make that safe to repeat,
but the originals are gone, so it is a choice and not a default.
