`apall` turns a two-dimensional frame into one-dimensional spectra. It is
four things, each also a task of its own, as in IRAF:
[apfind](task:apfind), [apresize](task:apresize), [aptrace](task:aptrace)
and [apsum](task:apsum), with [aprecenter](task:aprecenter) and
[apedit](task:apedit) beside them. They are apall with only their step on,
and hand the apertures to one another through `database/`:

1. **find** the spectra across the slit: the dispersion lines around `--line`
   (the middle, by default) are added, `--nsum` of them, so that a faint
   spectrum becomes a peak to centre; the `--nfind` strongest peaks at least
   `--minsep` apart are kept;
2. **resize** each aperture to where the profile, above a straight background
   between the minima either side, falls to `--ylevel` of its peak;
3. **trace** it along the dispersion every `--t-step` lines and fit a smooth
   curve (`--t-function`, `--t-order`): the spectrum does not sit on one
   column, because the camera distorts and the spectrograph flexes;
4. **extract**: add up the light inside the aperture at every line.

The port is IRAF's apextract, checked against IRAF's own output, and the
apertures and traces go to `database/ap<image>`, as in IRAF, where the next
frame can reuse them.

```console
$ harwood apall p/sp0019.fits sp0019.ms.fits --gain GAIN --readnoise RDNOISE \
      --nsum 20 --t-step 20 --weights variance
ok apall in 0.58s
  apertures: 1
  centres: 53.95
  limits: -2.90:+2.87
```

(The 1992 coudé-feed night of the course exercises; see
[the recipe](page:recipes/1992-spectroscopic-night).) `--explain` says what
the trace came to: 51 points, fitted to 0.017 pixels rms, drifting −1.51
pixels from one end of the spectrum to the other.

## Apertures from another frame

A frame that already has apertures in `database/` -- from an earlier run --
starts from them instead of finding the spectra again, as in IRAF. And
`--references` takes them from another frame: that is how an arc is
extracted, since a lamp filling the slit has no star to find or trace.

```console
$ harwood apall p/sp0018.fits sp0018.ms.fits --references sp0019 \
      --no-recenter --no-resize --no-trace --background none
```

The arc's lines are then measured on exactly the pixels the star's light was
summed from. Apertures taken from the database are recentred on this frame's
profile (`--recenter`, on by default): every aperture moves by the median of
their moves (`--shift`), so a small drift of the star on the slit is followed
and the apertures keep their spacing; `--no-shift` moves each by its own,
`--npeaks` uses only the brightest and `--aprecenter` only the apertures it
names. The log says what happened, in IRAF's words:
`RECENTER  - 2 apertures shifted by 1.18 for B.` `--references OLD` takes
only frames that have apertures of their own, `NEW<frame>` only those that do
not.

## Extraction

- **`--weights none`** (the default, as IRAF) is a plain sum, the end pixels
  by the fraction of their light inside the edge. IRAF gives no uncertainty
  for it, and neither does harwood.
- **`--weights variance`** is Horne's optimal extraction: the spatial profile
  is fitted along the dispersion and each pixel is weighted by how much of
  the light it holds and by its variance, so that the noisy edges stop
  diluting the bright middle. It needs the true `--gain` and `--readnoise`,
  and gives the uncertainty of every point. On a bright spectrum the noise
  is photons and there is little to recover; on a faint one, where read
  noise dominates the edges, there is a lot. The test suite measures it as
  scatter about a known answer: on a bright synthetic spectrum the plain sum
  scatters under one per cent more than the optimal one, and on one fifty
  times fainter, 48 per cent more. The 1992 frames, at a signal-to-noise of
  89 per pixel, are of the bright kind. `--clean` also replaces the pixels
  the profile says are cosmic rays.
- **`--background`** fits the sky across the slit in `--b-sample`, either
  side of the aperture, at every line, and subtracts it under the aperture.
  Leave a gap: the wings of the profile reach further than the aperture.

- **`--pfit fit2d`** fits the profile for variance weights as Marsh (1989)
  does, for a spectrum tilted across the pixel columns: overlapping
  triangles `--polysep` pixels apart across the aperture, each scaled by a
  polynomial of `--polyorder` terms along the dispersion, which follow the
  trace instead of smoothing each column. The default, `fit1d`, smooths each
  column along the dispersion (Horne), right while the trace stays within a
  column or so. With fewer, wider polynomials (`--polysep 0.7 --polyorder 4`)
  harwood gives IRAF's spectrum to 1e-4; with the defaults, a hundred unknowns
  solved in single precision, IRAF's answer depends on its rounding, and
  harwood's agrees with it to a fifth of each point's uncertainty.
- **`--nsubaps`** cuts each aperture across the slit into that many equal
  parts and extracts each as a spectrum of its own, with the same background
  and, for variance weights, the profile of the whole aperture: a way to see
  the light of a resolved source, or a slit, part by part. Aperture 2's
  parts are numbered 2, 1002, 2002…, as in IRAF, and each table's `SUBAP`
  card says which part it is; the parts add up to the whole. IRAF, with
  variance weights and a gain other than 1, leaves every part but the first
  in photons instead of counts; harwood gives them all in counts.

```warning
The gain and read noise must be the detector's. Guessed, every uncertainty
in the file is wrong by the same factor and nothing looks wrong: a gain of 3
where the header says 5.4 overstates them all.
```

`--edit` opens the profile with the apertures drawn on it, to set them by
hand with IRAF's apedit keys before extracting; `--fittrace` does the same
for the trace. `m` and `n` make an aperture, `f` finds them, `d` deletes the nearest,
`c`, `g` and `s` centre or move it, `l`, `u`, `y` and `z` set its limits, `b` opens the
background's icfit until its `q`, `i`, `j` and `o` number them, `a` is a switch that makes
the next ones act on all, and colon commands such as `:lower -5` and `:ylevel 0.2` work
as in IRAF. `w` and its second key set the window as IRAF's gtools does (`w x` zooms
about the cursor, `w f` flips the axis), and so do the `/` commands, `:/xwindow 10 60` or
`:/ytransform log`; the background's fit keeps its window from one aperture to the next,
as IRAF's does. Thirteen sessions IRAF ran say and draw the same, from its graphics
metacode. The mouse zooms besides, and `:.snap profile.png` saves the graph. Not ported:
`e` and `t`, the task going on to trace and extract, and `:parameters`, `:database`,
`:logfile`, `:plotfile`, `:read`, `:write` and `:image`.

## What it writes

One FITS file with a binary table per aperture, `sp0019.ms.fits[SPEC,1]`,
`[SPEC,2]`…, rather than IRAF's multispec image: the columns are `pixel`,
`flux`, `error`, `raw` (the unweighted sum), `sky` when there was a
background, `dq` when the frame has one, and `wavelength` once
[dispcor](task:dispcor) has run.

`error` comes from the frame's variance when it has one -- through the same
weights the flux was made with, the edge fractions of a plain sum or the
final weights of optimal extraction, plus the uncertainty of the sky that
was subtracted, propagated through the fit as skyeval makes it. So a plain
sum has an error too, which IRAF never gave, and it includes the flat and
the bias, which IRAF's noise model does not. Without a variance, `error` is
IRAF's sigma, for variance weights only. `dq` is each line's bits of the
pixels the aperture took, and dispcor carries it onto the wavelength grid. Every number is in the file, so nothing has
to be recomputed to read it; [wspectext](task:wspectext) writes it as text.

## IRAF's formats

`--format` writes the spectra as IRAF does instead, for taking them to IRAF,
PyRAF's `splot` or anything else that reads IRAF's spectra -- the same files,
by the same names, with the same header cards and pixels as IRAF's apall
writes them:

| `--format` | writes |
|---|---|
| `multispec` | `obj.ms.fits`: a line per aperture, and with `--extras` (the default, as in apall; apsum's is off) a plane per band behind it -- raw, background, sigma -- named by `BANDID1`, `BANDID2`… |
| `echelle` | `obj.ec.fits`, the same; with subapertures, one image per part, `obj1.ec.fits`, `obj2.ec.fits`… |
| `onedspec` | an image per aperture, `obj.0001.fits`, `obj.0002.fits`… (parts: `obj.1001.fits`) |
| `strip` | an image per aperture of its pixels straightened along the trace, the lower edge on the first line: the profile fitted times the spectrum with variance weights |

`APNUMn` gives each line's aperture, beam and limits across the slit, and the
world coordinates are IRAF's: pixels along the dispersion, counted from the
untrimmed frame for a trimmed one, or the frame's own linear WCS on that axis.
`--output` is the root name (`--output night1/obj` writes `night1/obj.0001.fits`).
harwood's other tasks read harwood's format, so a reduction that goes on in
harwood keeps the default.
