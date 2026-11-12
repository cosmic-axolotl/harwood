RVSAO's `emsao`: the redshift of a galaxy from its emission lines. RVSAO is the
Smithsonian Astrophysical Observatory's radial velocity package for IRAF (Mink &
Kurtz 1998), an external package rather than part of IRAF; it is what the course's
exercise 7.4 uses, and harwood carries it.

```console
$ harwood emsao ngc660_bg_sub.0002.fits --vel-corr none
$ harwood emsao l0020.ms.fits --specnum 1-3
$ harwood emsao galaxy.fits --vel-init guess --czguess 850 --logfiles emsao.log
```

What it does, as RVSAO 2.8.5 does it:

1. The spectrum is smoothed (`--nsmooth` passes of 1-2-1) and its continuum fit
   and subtracted (`--c-function spline3 --order 10`, clipped at
   `--s-low-reject`/`--s-high-reject` sigmas, RVSAO's contpars).
2. A first redshift: the tallest line near [O II] 3727, Hbeta, [O III] 5007 or
   Halpha (`--emsearch`), or `--czguess` as cz (`--vel-init guess`) or z (`zguess`).
3. Each line of `--emlines` (RVSAO's list, from Lyman gamma to [O II] 7330) where
   that redshift puts it, within `--wspan` Angstroms, kept when it stands
   `--linesig` times the rms of its continuum region above the continuum.
4. Around each, the unsmoothed spectrum fit with one, two or three Gaussians and a
   straight background: [N II]-Halpha-[N II] together, and the [S II] and
   [O III] pairs (`--emcombine`).
5. The redshifts of the lines weighted by the errors of their centres (plus
   `--disperr`); a line within 20 pixels of an end of the spectrum (X5, X6), with
   an equivalent width under `--lsmin` of its error (X4), or whose fit failed (X1),
   is listed and left out.
6. The velocity corrected to the solar system's barycentre (`--vel-corr
   barycentric`, the default) or the Sun (`heliocentric`), computed from the
   header -- DATE-OBS (yyyy-mm-dd or dd/mm/yy) or MJD-OBS, UT/UTOPEN/UTMID and
   EXPOSURE, RA, DEC, EPOCH or EQUINOX, SITELONG/SITELAT/SITEELEV or OBS-* or
   OBSERVAT -- with Stumpff's BARVEL; `file` and `hfile` take the BCV or HCV card.
   When any of them is missing the correction is 0, as RVSAO's is.

The report goes to the terminal and to `emsao.log`, appended (`--logfiles`), in
RVSAO's form: the combined velocity, the emission velocity with its error and how
many of the lines found were used, and a line for each line: rest and observed
wavelength, pixel, z, velocity, its error, equivalent width and weight.

RVSAO's own defects are kept, because they decide the numbers: among them, a
spectrum without RA and DEC is placed at 0h 0d -- but only the first of a run (the
next ones say "No position") -- and `--wspan`'s fraction is dropped.

Not ported yet: the graph and its cursor (`displot`, `curmode`), saving the
velocity to the header (`save_vel`), the sky spectrum, the archive and report
modes other than 1. On 44 cases -- the NGC 660 nucleus, galaxies at z = 0.05 and
0.3, one line, none, log-linear and multispec spectra, every velocity correction,
the initial guesses and every parameter of the search, fit and continuum -- the log
is RVSAO 2.8.5's, compiled against IRAF 2.18, line for line.
