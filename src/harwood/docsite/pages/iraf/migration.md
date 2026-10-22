# Coming from IRAF

The tasks keep their names, their parameters keep theirs, and they mean the
same things: `ccdproc` with `zerocor` and `flatcor`, `imcombine` with
`reject=ccdclip` and `lsigma`, `phot` with `apertures`, `annulus` and
`epadu`. Coordinates are 1-based, (x, y), as in IRAF and DS9. Image
sections are written the IRAF way, `obj.fits[1:100,*]`, and lists are
`@obj.lis`. The ported tasks are checked against IRAF 2.18's own output in
the test suite, number by number.

What changes is the cl around them.

## The cl, and what to do instead

| in IRAF | in harwood |
|---|---|
| `noao`, `digiphot`, `apphot` | nothing to load: every task is always there. `harwood list apphot` shows a package; the window's Tasks panel walks the tree |
| `bye` | nothing to unload; in the Tasks panel, *◀ Back* |
| `epar daofind` | `harwood daofind --editpar`: every parameter explained, saved into the project |
| `lpar daofind` | `harwood lpar daofind`, which also says where each value came from |
| `daofind.fwhmpsf = 3` | `fwhmpsf: 3` under `daofind:` in `harwood.yaml`, or `--fwhmpsf 3` for one run |
| `unlearn daofind` | take `daofind:` out of `harwood.yaml`; in `--editpar`, clearing a value puts the default back |
| `dpar daofind` | `harwood log --replay N`: the exact command of run N |
| `help daofind` | `harwood help daofind`, or the task's page here (F1 in the window) |
| `uparm/`, `login.cl` | nothing in your home directory: the night's `harwood.yaml` holds its parameters |
| `display obj 1` | `harwood display obj.fits --frame 1`, in harwood's own window; no ds9 needed |
| `imexamine` | the same keys -- `r`, `a`, `m`, `l`, `c`, `e`, `s`, `:radius 7` -- in the window |
| `!ls` | `!ls` at the `hw>` prompt |
| `history`, `^` | `history`, the up arrow, Ctrl+R |
| `logout` | `exit` |
| `cl < night.cl` | a Python script calling `harwood.run`, or a shell script ([Python](page:concepts/python)) |
| `ccdproc` with `output=""` (in place) | a new directory per run, unless `--inplace` ([outputs](page:concepts/outputs)) |

## Images go both ways

A harwood image is three planes in one FITS file: `SCI`, the data; `VAR`,
its variance; and `DQ`, a number per pixel saying why it should not be
trusted -- saturated, a cosmic ray, a bad column -- in the bits DRAGONS uses.
IRAF keeps one number per pixel, so the two are not the same file, and
harwood is careful that neither direction is a dead end:

- **From IRAF.** Any IRAF image is read as it is. `obj.fits` with one HDU is
  the data; its `BPM` mask becomes the DQ. There is no variance until a task
  that knows the gain makes one -- harwood never invents a gain.
- **To IRAF.** **`--format iraf`** writes the data alone, in one HDU, as
  IRAF reads it, and the DQ as a FITS mask named by `BPM` that IRAF's
  `ccdproc`, `fixpix` and `imcombine` use. The run says what stayed behind --
  the variance -- rather than dropping it quietly. `ccdproc`, `imcombine`
  with its presets, `imarith`, `cosmicrays` and `fixpix` have it now; the
  other tasks that write
  images get it as they learn to carry the variance (design contract,
  section 12.2), and until then write one HDU anyway.
- **Halfway.** IRAF opens a harwood file directly: `imstat obj.fits[SCI]`,
  `hselect obj.fits[SCI] OBJECT,EXPTIME yes`. The extensions carry
  `INHERIT = T`, so the exposure's keywords are there too.
- **For a whole reduction** that goes back to IRAF, say it once in
  `harwood.yaml`: `params: {all: {format: iraf}}`. The same place takes
  `uncertainty: ERR` for a standard deviation instead of a variance, or
  `UNCERT` for astropy's `CCDData` layout.

| in IRAF | in harwood |
|---|---|
| `obj.fits` | `obj.fits` (reads `SCI`), or `obj.fits[SCI]`, `[VAR]`, `[DQ]` |
| `BPM` keyword, `.pl` mask | the `DQ` plane; `--format iraf` writes it back as a `BPM` mask |
| `ccdproc` | the same steps, and the variance from `GAIN` and `RDNOISE` |
| results to take back | `--format iraf` on the task that writes them |
| `phot`'s `merr` | from the image's `VAR` when it has one; IRAF's own, from `epadu` and the sky, as `merr_model` |
| `apall`'s sigma band | `error`, from the `VAR` when there is one -- for a plain sum too |

## Things harwood does that IRAF did not

- `--plan` and `--explain` on every task: the steps with real values, and
  why, before anything is done.
- A `run.yaml` beside every result and a log that can repeat any run.
- Every warning with what to do about it.
- Tasks IRAF does not have yet in harwood get an answer: `harwood help
  splot` says whether it is planned, replaced, or not ported and why.

## Every IRAF task, and what became of it

All of IRAF 2.18's tasks, by name. *in harwood* links to the task's page;
*replaced* says what does it now; *not ported* says why.

```generated
migration-table
```
