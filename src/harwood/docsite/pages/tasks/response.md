The response of a long-slit flat, as IRAF's `longslit.response` makes it: the
calibration image (the dome flat) divided by a smooth fit to the spectrum of
the normalization image (usually the flat itself) averaged across the slit.
What is left is close to 1 -- the pixel-to-pixel response and the slit's
profile, with the lamp's colour and the spectrograph's response along the
dispersion taken out.

```console
$ harwood response Flat.fits Flat.fits Resp.fits --dispaxis 2
$ harwood response flat_btp.fits flat_btp.fits flat_response.fits --function spline3 --order 5 --low-reject 3 --high-reject 3
$ harwood response Flat.fits Flat.fits Resp.fits --interactive
```

The fit is ICFIT's: `--function`, `--order`, `--sample`, `--naverage`, the
rejection `--low-reject`, `--high-reject`, `--niterate` and `--grow`, from IRAF's
defaults (spline3 of order 1, no rejection). `--interactive` opens the icfit
window on the spectrum, with IRAF's keys and colon commands (`:function cheby`,
`:order 5`, `f`, `q`); `--cursor` reads the same keys from a file, a fit up to
its `q`. The dispersion axis is the DISPAXIS keyword, or `--dispaxis`.

`--threshold` sets the response to 1 wherever the fit or the pixel itself is
below it -- the unlit ends of a slit. A calibration given as a section is fitted
and divided as a section: a new response image is the size of the whole image,
ones outside the section, and one that already exists is written only where the
section is, as IRAF builds a response from several parts of a slit.

The response carries CCDMEAN = 1, as IRAF's does, with the CCDMEANT it copied
from its input -- older than itself, so that [ccdproc](task:ccdproc) computes
its mean again before dividing by it, as IRAF's ccdproc does.

On the course's 6.1 frames harwood's response of Flat is IRAF's Resp.fits pixel
for pixel -- with response's defaults: the session that made that Resp kept
spline3 of order 1, not the chebyshev of order 5 the text asks for. On 7.1,
`--function spline3 --order 5 --low-reject 3 --high-reject 3` gives
flat_response.fits. Every case compared on synthetic flats -- each function,
rejection, averaging, sample ranges, a threshold, both dispersion axes,
sections and an existing image -- gives IRAF 2.18's file pixel for pixel.
