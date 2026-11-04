`reidentify` carries a solution [identify](task:identify) made on one
spectrum to others. Each line of the reference is put where the reference's
solution says its wavelength now falls, centred on the new spectrum, and the
solution refitted from the lines found. Three jobs are done this way:

- **The other apertures of a multi-object arc.** They share the lines, not
  the dispersion: identify one aperture by hand and carry it to the rest.
- **Along a long slit.** A two-dimensional arc is reidentified every
  `--step` lines (each the sum of `--nsum`), which is how the curvature of
  the lines across the slit is measured.
- **The night's other arcs.** Name them after the reference, and each
  aperture or line is reidentified from the reference's own solution for it.

```console
$ harwood reidentify arc3 arc3,arc3b --coordlist 'linelists$thar.dat'
$ harwood reidentify arc2d arc2d --step 10 --nsum 5
```

The solutions go to `database/id<image>`, where [dispcor](task:dispcor) finds
them.

## Reading the log

One line per spectrum, as IRAF prints it:

```text
          Image Data    Found     Fit Pix Shift  User Shift  Z Shift      RMS
arc3 - Ap 1           103/109  61/103      1.55       0.446  1.07E-4    0.036
```

**Found** is how many of the reference's lines were centred again,
**Fit** how many of those the fit kept, then the mean shift in pixels and in
wavelength, and the fit's rms. A spectrum that loses many lines, or whose rms
stands out from its neighbours', is the one to look at. Along the reference
(the other apertures or lines of the same arc), losing more than `--nlost`
lines refuses the solution; with `--trace` it also stops the tracing in that
direction, since the next line would start from a bad one.

## The shift

The reference's lines are first moved by `--shift`, in wavelength, before
they are centred. On a long slit without `--trace` each line starts from the
shift measured on the line before, plus `--shift` per step -- the lines drift
smoothly across the slit, so that is a good guess. With `--trace` each line
starts from the previous line's solution instead of the reference's.
`--shift INDEF --search INDEF` finds the shift from the spectrum's own peaks,
for an arc that has moved by more than the centring radius. `--shift INDEF`
with a radius, `--search 5` (Angstroms; negative, a fraction of the
spectrum's range), searches for it as autoidentify does: the pattern of the
line list's lines within that distance of the reference's solution, with the
same dispersion -- for an arc that has moved by more than its peaks can say.

`--addfeatures` adds the lines of `--coordlist` the solution finds on each
spectrum, and refits.

## Not yet

IRAF's interactive mode, which stops at each fit, is not ported: only
`--interactive no` runs. Everything else is IRAF's t_reidentify.x, and checked
against IRAF's own output on nine sets of parameters, where the log comes out
the same character for character, and three with a search radius, where the
same lines are found and rejected and the log agrees to within one in its
last figure.
