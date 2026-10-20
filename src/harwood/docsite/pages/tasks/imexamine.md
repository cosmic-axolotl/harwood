Point at the image and press a key; the answer goes to the Examine panel
and the session log, and to `--logfile` when one is named.

| key | | key | |
|---|---|---|---|
| `r` | radial profile and FWHM | `j` | Gaussian fit along the line |
| `a` | aperture photometry and FWHM | `k` | Gaussian fit along the column |
| `m` | statistics of a box | `h` | histogram of a box |
| `l` | plot the line | `e` | contour plot of a box |
| `c` | plot the column | `s` | surface plot of a box |
| `x` | coordinates and value | `z` | the 11 × 11 pixels around it |
| `n`, `p` | next and previous frame | `?` | list the keys |
| `q` | end the task | `:` | a colon command: `:radius 7` sets, `:radius` shows |

Every star is recentred before it is measured, so the cursor only has to
land near it. The parameters are IRAF's, with IRAF's defaults and meanings:
`radius` is the aperture, `rplot` how far the profile is drawn, and `r` and
`a` widen the aperture to three times the direct FWHM, `iterations` times.
Every key's arithmetic is a port of IRAF's own, and the tests hold `r`,
`a`, `m`, `j`, `k` and `z` to what IRAF 2.18 printed for the same pixels.

## Without a screen

Put the positions in a cursor file -- one `x y [frame] [key]` per line --
and pass it as `--imagecur`: the same code answers, so the numbers are the
ones the window would have shown.

```example
title: The profiles of the two brightest stars
after: reduced
run: harwood imexamine red/obj001.fits --imagecur stars.cur
output: |
  MFWHM=2.95  direct=7.46
  MFWHM=3.01  direct=2.93
```

## Reading the line

The line reads as IRAF's: radius, magnitude, flux, sky, peak, ellipticity,
position angle, Moffat beta, and IRAF's three FWHMs -- of a fit to the
enclosed flux, of a fit to the pixels (`MFWHM` for a Moffat, `GFWHM` for a
Gaussian), and the direct one, measured without a model. The example shows
why there are three: the second star's are all about 3, the FWHM the sample
night was made with; the first star has a fainter neighbour seven pixels
away, inside its profile, and while the fit still says 2.95 the direct
measure, which takes the light as it comes, says 7.46. Two measures that
disagree are the sign to look at the star.

Colon commands are IRAF's: a name shared by several of IRAF's psets
(`width`, `rplot`, `naverage`, `ncolumns`, `floor`…) changes the one of the
last graph drawn, and `:unlearn` resets it. In a cursor file they go after
the frame, `612 540 1 :radius 7`, and act on the lines that follow.
