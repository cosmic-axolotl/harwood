RVSAO's `emsao`: the redshift of a galaxy from its emission lines. RVSAO is the
Smithsonian Astrophysical Observatory's radial velocity package for IRAF (Mink &
Kurtz 1998), an external package rather than part of IRAF; it is what the course's
exercise 7.4 uses, and harwood carries it.

```console
$ harwood emsao ngc660_bg_sub.0002.fits --vel-corr none
$ harwood emsao l0020.ms.fits --specnum 1-3
$ harwood emsao galaxy.fits --vel-init guess --czguess 850 --logfiles emsao.log
$ harwood emsao ngc660_bg_sub.0002.fits --vel-corr none --displot --save-vel
$ harwood emsao ngc.fits --cursor keys.txt --answers answers.txt --save-vel
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

## The graph and its keys

`--displot` opens RVSAO's graph in the window: the spectrum (smoothed as for the
search), each emission line of the list marked where the velocity puts it -- a solid
line over a line found and fit, a dashed one where none was found or `-` dropped it
-- and the absorption lines (`--ablines`) below; beside it, RVSAO's summary: the
date, the correction, VELOCITY with its quality, the correlation's velocity and each
template's when xcsao wrote them to the header, the emission lines' velocity, and
each line found. `--dispmode 3` gives the graph the whole screen; 4 and 5 show the
spectrum with its continuum subtracted. `--vel-plot` chooses the velocity the
lines are marked at, `--dispem`/`--dispabs` which lines are marked.

The keys are RVSAO's (`?` lists them):

| key | what it does |
|---|---|
| `z` `z`, `u`, `.` | zoom between two positions, unzoom, cancel |
| `/`, `h`, `k`, `o` | whole screen or with the summary; the heading; the continuum; which lines are labelled |
| `v` | mark the lines at the emission, correlation, combined or guessed velocity |
| `e`, `a` | the velocity from the emission (or absorption) line under the cursor, named (`ha`, `OIII`) or by its rest wavelength |
| `s` | set VELOCITY: from the emission lines, the correlation, both, the line set with `e`/`a`, or typed |
| `y`, `n`, `j` | its quality: good, bad, questionable |
| `+`, `-` | keep or drop the line found nearest the cursor, then `f` |
| `b`, `r`, `g`, `l`, `m`, `i`, `c` | the search's blue and red limits, smoothing, line search parameters, the fit's smoothing, the initial velocity, the continuum's order |
| `f` | fit again with what the keys changed |
| `d` `d`, `,` | replace the spectrum between two positions by a straight line; set one pixel to the cursor |
| `t` | the correlation velocity from another template |
| `(`, `)` | the previous or next aperture of a multispec |
| `@` | a picture of the graph (an SVG; `--hardcopy` writes one of every graph) |
| `q`, `]`, `x` | go on (with any refit the keys asked for), go on without refitting, go on forgetting the edits |

`--cursor` reads the keys from a file instead (IRAF's cursor file, `x y 1 key` a
line) and `--answers` what their questions read, a line each. `--save-vel` writes
the velocity into the header as RVSAO does -- VELOCITY, Z, CZERR, VELQUAL, BCV, the
line a velocity was set from (VELSET), CZEM and a card per line found (EMLINEn; a
multispec's APVELn, APVEMn and ELm_n), the last EMSAO card kept as HISTORY -- and the
spectrum the keys edited goes back into its file.

RVSAO's behaviour is kept where it decides what happens. `q` is the cursor's end of
file to RVSAO (gtools' `gt_gcur`): it leaves the graph with any refit `e`, `v`, `+`
or `b` asked for still asked for, and the fit runs again; `]` goes on without it.
`?` lists the keys and leaves the graph. The order `c` sets is for this spectrum
only. A spectrum edited is written when the graph is left, by `]` too; only `x`
forgets it.

Where RVSAO fails, harwood does what its code means to do. Without `save_vel` RVSAO
opens the spectrum read-only, so `y`, `n`, `j` and `s` end in "Cannot update image
header"; harwood writes the quality (RVSAO's qwhead: VELOCITY, VELQUAL and QPLOT)
and the lines' cards. `c` closes the graph and never opens it again, so the next
graph drawn ends emsao; here the graph stays. `d`'s second press calls fillpix
without its last argument and is a segmentation violation, always; here it fills.
RVSAO's FITS kernel writes back the header it holds in memory, RA, DEC, HA and ST of
00:00:00 and today's DATE among them, and the next spectrum then has a position and
a date; harwood writes RVSAO's own cards only. `#` (debug) is not ported.

`--report-mode` gives RVSAO's other forms: 0 the normal one with the equivalent
widths' errors, 10 without the heading; 2 to 9 and 11 a line per spectrum (the
velocities, each line's velocity, shift in Angstroms or pixels, height, width and
equivalent width, the instrument, the position and its altitude and azimuth), 12 a
commented table; negative, the same tab-separated under column heads. RVSAO's own
slips are kept: mode 2's combined error is the correlation's, modes 4 and 8 take
the sine of the latitude's degrees, mode 7's format has a field more than its
values (and fmtio's warnings follow). `--skynum` names the sky's aperture in a
multispec, one for each spectrum in turn: its level beside each line enters the
error of its equivalent width, and so which lines `--lsmin` keeps (RVSAO's getsky
also prints getimage's debugging lines, reading an argument it is not given;
harwood reads the sky quietly).

`archive` is not ported: its records are SAO's own binary format, read only by
SAO's pipeline.

On 81 cases -- the NGC 660 nucleus, galaxies at z = 0.05 and 0.3, one line, none,
log-linear and multispec spectra, every velocity correction, the initial guesses,
every parameter of the search, fit and continuum, every report mode and the sky --
the log is RVSAO 2.8.5's,
compiled against IRAF 2.18, line for line; on 112 sessions of the graph -- every
key, a red-to-blue spectrum, a multispec, a header xcsao wrote -- what it prints,
writes to the header and the pixels, and draws (the window, the spectrum, every mark
and label, the summary) is RVSAO's.
