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
$ harwood apall p/sp0019.fits sp0019.ms.fits --gain 5.4 --readnoise 5 \
      --nsum 20 --t-step 20 --weights variance
ok apall in 0.24s
  apertures: 1
  centres: 53.95
  limits: -2.90:+2.87
```

(The 1992 coudé-feed night of the course exercises; see
[the recipe](page:recipes/1992-spectroscopic-night).) `--explain` says what
the trace came to: 51 points, fitted to 0.017 pixels rms, drifting −1.51
pixels from one end of the spectrum to the other.

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
  88 per pixel, are of the bright kind. `--clean` also replaces the pixels
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
for the trace.

## What it writes

One FITS file with a binary table per aperture, `sp0019.ms.fits[SPEC,1]`,
`[SPEC,2]`…, rather than IRAF's multispec image: the columns are `pixel`,
`flux`, `error` (variance weights only), `raw` (the unweighted sum), `sky`
when there was a background, and `wavelength` once
[dispcor](task:dispcor) has run. Every number is in the file, so nothing has
to be recomputed to read it; [wspectext](task:wspectext) writes it as text.
