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

## The graph and its keys

`--displot` opens RVSAO's graph in the window: above, the spectrum (smoothed
`--nsmooth` times) with the range correlated marked by dashed lines; below, the
correlation with the best template around its peak (`--cvel`, `--dvel`), the points
at `-pkfrac` of its height marked when `--pkfrac` is negative; at the right, RVSAO's
summary: the date, the correction, the range, the grid and the filter, each
template's velocity and peak, and each one's velocity, error and R. `--dispmode 0`
shows only the range correlated, `-1` the spectrum from zero; `2` and `3` (and the
`l` key, and `m` to 2 or 3) open emsao's graph of the spectrum first -- the
emission lines emsao wrote to the header marked as found, the rest of
`--emlines` and `--ablines` where the correlation's velocity (`--vel-plot`) puts
them, with all of emsao's keys (see emsao); `x` there comes back to the
correlation, `f` correlates again.

The keys are RVSAO's (`?` lists them): `r` redraws; `t` takes another template's
velocity (by its place in R); `g` smooths the spectrum (shown at the next `r`); `m`
changes the display mode; `c` the peak's fit (mode, fraction, the filter) and `v`
the velocities a peak may have, and correlate again; `f` correlates again; `y`, `n`
and `j` set the quality (VELQUAL, with `--save-vel`); `@` writes a picture (an
SVG; `--hardcopy` writes one of every graph); `x` goes on without writing the
header; `q`, and any key it does not know, goes on. `--cursor` reads the keys from
a file instead (`x y 1 key` a line) and `--answers` what their questions read.

RVSAO's own `c` and `v` read what they ask, ask for a new correlation and lose it:
xcfit reads `pkfrac`, `minvel` and `maxvel` from the parameters again, and the
filter from its own copies, so that only the peak's mode changes anything. Here
they all reach the correlation, which is then RVSAO's with those parameters from
the start. Without a date of observation the summary says "No observation date"
twice, as RVSAO's does.

RVSAO's `d` in the editor calls fillpix without its last argument and ends in a
segmentation violation; here it fills. `--correlate no` correlates nothing: it
reports and draws what an xcsao before wrote to the header, to look at it again or
set its quality (`y`, `n`, `j`, written with `--save-vel` as RVSAO's qwhead: VELQUAL
and QPLOT), with what the header does not hold left as RVSAO leaves it (0 points,
the peak at 0, fit 0.0, +tc). Not ported: `archive`, whose records (.ansum,
.corr) are SAO's own binary format, read only by SAO's pipeline. On 135 cases -- one template or several, a
broadened galaxy, noise, zero padding, every limit in wavelength, the initial
velocity and its passes, a velocity window, the peak's fraction or points and the
cosine fit, the filter, the taper, the grid's size, lines clipped from the spectrum
or the template, zeroed ends, renormalisation, log-linear and multispec spectra,
every velocity correction, the template's own velocity and flags, the correlation
in wavelength and pixels, every report mode and save_vel -- the log and the header
are RVSAO 2.8.5's, compiled against IRAF 2.18, line for line; on 70 sessions of the
graph, emsao's graph and the editor from it among them, what it prints and draws is RVSAO's.
