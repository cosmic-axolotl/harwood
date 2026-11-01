Takes the sky off a long-slit frame as one spectrum (Kelson 2003, PASP 115, 688). IRAF's `background`,
and what [apall](task:apall) does, fit each line across the slit, so that a sky line leaning across
the detector is cut differently by each line, and what is left of it is the sharp residual everyone
knows. Here the lean is what helps. Each pixel of the slit samples the sky's spectrum at its own place,
a fraction of a pixel from its neighbours', and a spline in that one coordinate sees it far better
than any line does. It is evaluated back at each pixel, so nothing is resampled and what remains has
the noise of the pixel.

```console
$ harwood skyspline obj.fits obj.sub.fits --sky obj.sky.fits --tilt-frame arc.fits --dispaxis 2
```

**What it does.** The tilt starts from `--tilt-frame`, an arc, or the image's own features: bands
across the slit against the middle, one tilt all along the dispersion. A tenth of a pixel of tilt
leaves several sigma on a sharp line, and no correlation gets closer than that, so the sky's own spline
says the rest: where the residual follows the spline's slope, the tilt is off, and a polynomial of
`--xorder` and `--yorder` takes it up. The slit, `--slit`, and what has an object on it, `--object`,
come from the profile if not given, as columns from 1; a frame whose sky is faint against its star wants
`--slit`. The sky and the slit's illumination, a polynomial of `--order`, are fitted in turn, with `--clip`
sigma of the noise by `--gain` and `--rdnoise` keeping cosmic rays and an object's wings out. The frame
is left in the input's own pixels, with zero sky off the slit. A pixel flagged in the DQ stays out of
the fit.

`--spacing` is the spline's knots, in pixels. Left out, it is 0.5 when the lines lean by 2 pixels or
more across the slit, as they then sample that finely, and 1 when not, where it comes to a fit by
lines.

**When it helps.** It takes tilt. It says how much there is, and with under a pixel across the slit it
says that this is no better than a fit by lines. It also says what remains of the sky against its noise,
and warns above 1.5: usually a flat that does not take the pixels out. And it says when a frame has
hardly any sky to take off.

Not IRAF's, so there is no IRAF to compare with. It is held to skies put there beforehand: sharp lines
leaning 3 pixels across the slit, an object on the slit, cosmic rays, an illumination that falls off,
where what remains is the noise and the tilt comes back to a few hundredths of a pixel, and a fit by
lines leaves several times that. On the tutorial's real twilights there is little to gain, as the lines
of that spectrograph lean by 0.04 pixels across the slit: with a rough flat it leaves 1.14, 1.38 and 1.34
of the noise, and a median by lines 1.16, 1.56 and 1.52. The method's real gain, on real tilted lines,
is not shown here: the tutorial's frames have none, and its objects have hardly any sky.
