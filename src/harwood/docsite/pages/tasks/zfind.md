A redshift the way today's surveys measure them (SDSS's spectro1d, DESI's Redrock):
by asking, at every trial redshift, how well the whole spectrum is explained, and
taking the best over the whole range. harwood's own; IRAF and RVSAO have nothing
like it.

```console
$ harwood zfind ngc660_bg_sub.0002.fits
$ harwood zfind galaxy.fits --zmax 0.8 --device chi2.svg
$ harwood zfind star.fits --lines "" --templates k0iii.fits --zmin -0.002 --zmax 0.002
```

At each z (steps of `--dv` km/s in ln(1+z), from `--zmin` to `--zmax`) the model is
a Legendre continuum (`--npoly` terms) plus a Gaussian at every line of the list
that falls in the spectrum -- RVSAO's emission list by default, or `--lines`, a
file of rest wavelengths -- each with its own amplitude, which cannot be negative,
fit by linear least squares; template spectra (`--templates`, rest frame) can join
or replace the lines, for absorption-line spectra. Four line widths are tried and
the best refined.

The answer is the lowest minimum of chi2(z), refined by a parabola; its error is
where chi2 has risen by 1, the noise scaled so that the best fit's reduced chi2 is
1 when the spectrum has no errors of its own (harwood's spectra use their error
column). `deltachi2`, the next distinct minimum (more than `--separation` km/s
away) minus the best, says how sure it is: under 9 the redshift is flagged as
doubtful, and the next candidates are listed with their chi2.

Why it is here: emsao decides from one line -- the tallest near Halpha is taken as
Halpha -- and in the NGC 660 nucleus of the course [N II] 6583 is the taller, so
its search answers cz = 1704 km/s. zfind fits all the lines at once and gives
865 km/s, with deltachi2 2250, as emsao does when it is told where to start.

Held to what is known: spectra made at z = 0.0026, 0.05 and 0.3 recovered within
their errors, the errors' pull over many noise draws spread by 1.02, noise alone
flagged, a template's absorption redshift recovered.
