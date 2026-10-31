`apall` turns a two-dimensional frame into one-dimensional spectra. It is
four things, which IRAF also offers apart as `apfind`, `apresize`,
`aptrace` and `apsum`:

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
as in IRAF. Ten sessions IRAF ran say and draw the same, from its graphics metacode. Not
ported: `w` and the `/` commands (the mouse zooms), `e` and `t`, the task going on to trace
and extract, and `:parameters`, `:database`, `:logfile`, `:plotfile`, `:read`, `:write`
and `:image`.

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
