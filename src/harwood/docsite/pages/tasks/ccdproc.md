`ccdproc` is the spine of an imaging reduction: it takes raw frames and
applies, in IRAF's order, each correction you switch on --

| step | switch | what it does |
|---|---|---|
| fixpix | `--fixpix` | replaces the bad pixels of a mask (`--bpm`) by interpolating across them |
| overscan | `--overscan` | fits the overscan strip (`BIASSEC`) along the readout and subtracts the fit |
| trim | `--trim` | keeps the illuminated part (`TRIMSEC`) |
| zero | `--zerocor` | subtracts the master bias (`--zero`) |
| dark | `--darkcor` | subtracts the master dark, scaled by exposure time (`--dark`) |
| flat | `--flatcor` | divides by the master flat (`--flat`), normalised to one |

The arithmetic is IRAF's ccdproc, ported, and checked against IRAF's output
on real frames. The overscan fit is IRAF's icfit -- a Legendre, Chebyshev
or spline of `--order` terms, with rejection -- and `--interactive` opens it
in a panel to fit by hand, with the parameters you settle on written to the
log so the run can be repeated in batch.

```example
title: A whole night's object frames, after the calibrations
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits --output red
output: |
  -> red/obj001.fits
```

## It reads the headers first

`--plan` shows the steps against the actual headers: which strip is the
overscan, which bias and flat it will use, and which steps it will skip.

```example
title: What it would do, and why
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits --plan --explain
output: |
  The overscan is a strip of pixels the detector reads out without exposing
```

**It never does a step twice.** Each step leaves a mark in the frame's
header (`OVERSCAN`, `TRIM`, `ZEROCOR`, `DARKCOR`, `FLATCOR`, `FIXPIX`), and a
step whose mark is there is skipped -- the bookkeeping that keeps a night
from having its bias subtracted twice, silently and for good.

## The calibrations can be found, not named

Give the bias and the flat among the inputs, without `--zero` and
`--flat`, and ccdproc picks them out by their image type, processes them
first if they need it, and gives each frame the flat of its own filter --
the *subset*, in IRAF's word, because it need not be a filter. `--ccdtype
object` says which frames to correct; the rest are only there to be found.

```example
title: Calibrations found by type, flats matched by filter
after: flat
run: harwood ccdproc @obj.lis Zero.fits FlatV.fits --overscan --trim --zerocor --flatcor --ccdtype object --output red --plan
output: |
  find zero: Zero.fits
  find flat: V -> FlatV.fits
```

The first time a subset is met, a line for it is added to the `subsets`
file in the reduction directory: the left column is what the header says,
the right one the short name harwood puts in file names. Edit the right
column -- `B`, `V` -- and nothing else needs to change.

## Where it writes

Without `--output`, into a directory of its own under `output/` with the
frames keeping their names; with a directory, there; with one image, to
that name; with `%old%new%`, side by side. `--inplace` corrects the inputs
themselves, as IRAF does by default -- the marks make that safe to repeat,
but the originals are gone. A plain text line per frame per step goes to
`harwood.log` (`--logfile`), to watch in another window.

## Headers from other instruments

The keywords are read through an instrument translation: which keyword
holds the filter, what each `IMAGETYP` value means. See
[setinstrument](task:setinstrument) and [ccdlist](task:ccdlist), which is the
check to run first: a frame it calls `unknown` is one ccdproc will not know
what to do with either.
