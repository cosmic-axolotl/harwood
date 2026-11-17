RVSAO's `xcsao`: the radial velocity of a star or a galaxy by cross-correlating
its spectrum with templates (Tonry & Davis 1979). RVSAO is the Smithsonian
Astrophysical Observatory's radial velocity package for IRAF (Mink & Kurtz 1998),
an external package rather than part of IRAF; harwood carries it, beside
[emsao](task:emsao), which takes a redshift from emission lines.

```console
$ harwood xcsao star.fits --templates hd12345.fits --svel-corr none
$ harwood xcsao galaxy.fits --templates 'g2.fits,k1.fits,a1.fits' --tempdir templates/
$ harwood xcsao galaxy.fits --templates g2.fits --vel-init guess --czguess 6000 --nzpass 1
```

What it does, as RVSAO 2.8.5 does it, for each spectrum and each template:

1. Both rebinned onto one grid even in log wavelength -- `--ncols` points (a power
   of two) over the range they share, or `--st-lambda`..`--end-lambda`, or the
   template's own `name:w1-w2` -- so that a velocity is a shift by the same number
   of bins all along. With `--vel-init guess` (cz) or `zguess` (z), the template
   is shifted to `--czguess` first, and `--nzpass` more times to what it finds.
2. Each one's continuum fit and taken off (RVSAO's contpars: `--c-function
   spline3 --order 10`, clipped at `--s-low-reject`/`--s-high-reject` sigmas for
   the spectrum, `--t-*` for the template); with `--s-emchop yes` the spectrum's
   emission lines above `--s-em-reject` sigmas (or absorption below
   `--s-abs-reject`) replaced by the continuum, `--t-emchop` the template's; the
   ends tapered by a cosine bell over `--bell-window` of the points.
3. Both transformed, filtered -- a cosine taper up from `--low-bin` to
   `--top-low`, flat to `--top-nrun`, down to `--nrun`: the continuum's slow
   wiggles and the pixel noise out -- multiplied and transformed back: the
   cross-correlation, normalised by the two spectra's rms.
4. Its highest peak between `--minvel` and `--maxvel` fit by a parabola above
   `--pkfrac` of its height (above 1: that many points), or by a quartic
   (`--pkmode 2`). Its centre is the
   velocity; R is its height over twice the rms of the correlation's
   antisymmetric part, and the error is 3w/8(1+R) of its width w.
5. The velocity corrected for the template's own velocity (VELOCITY in its
   header), its TSHIFT and `--tshift`, and to the barycentre (`--svel-corr`, from
   the header as for emsao; `--tvel-corr` for the template). The template with
   the highest R is the best.

A template's header can ask for its own treatment, as RVSAO reads it: SUBCONT = F
(its continuum is already off), DIVCONT (divide rather than subtract), FI-FLAG
(already filtered), OVERLAP (correlate the whole overlap), ZEROPAD, PEAKFRAC,
CHOPEM.

The report goes to the terminal and to `xcsao.log`, appended (`--logfiles`), in
RVSAO's form: the spectrum, its position and date, the range correlated, the
filter, and for each template its velocity, the peak, its height, R, the velocity
and its error -- or, with `--report-mode 2`, a line for each spectrum and
template: R, the velocity, its error, the peak's height and width.

Not ported yet: the graph and its cursor (`displot`, `curmode`), saving the
velocity to the header (`save_vel`), the archive, `--correlate wavelength` or
`pixel`, the cosine peak fit (`pkmode 3`) and report modes other than 1 and 2. On
55 cases -- one template or several, a broadened
galaxy, noise, zero padding, every limit in wavelength, the initial velocity and
its passes, a velocity window, the peak's fraction or points, the filter, the
taper, the grid's size, lines clipped from the spectrum or the template, zeroed
ends, renormalisation, log-linear and multispec spectra, every velocity
correction, the template's own velocity and flags -- the log is RVSAO 2.8.5's,
compiled against IRAF 2.18, line for line.
