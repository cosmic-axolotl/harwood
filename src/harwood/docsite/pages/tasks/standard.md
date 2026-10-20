Measures a flux standard -- a star whose flux somebody has already measured
and tabulated in bands -- through this instrument: its wavelength-calibrated
spectrum is integrated over each tabulated band and divided by the band's
width and the exposure time, giving counts per second per Ångström beside
the star's true flux in the same band.

```console
$ harwood standard std1.w.fits std --star-name bd284211 --caldir onedstds$spec50cal
```

- The star's table comes from IRAF's `onedstds$` directories (`spec50cal`,
  `spec16cal`, `irscal`…), which ship with harwood unchanged: the standards
  are somebody else's measurements, and reproducing them would only
  introduce errors. No IRAF is needed.
  `harwood standards --star bd284211` shows every directory that has the
  star -- the same star often appears in several, with different band
  widths and sometimes different fluxes, because they were calibrated
  against different primary standards. Which to use is a real decision.
- A band only partly covered by the spectrum is dropped, not scaled up: a
  half-covered band looks exactly like a star half as bright, and afterwards
  nothing could tell the two apart.
- The airmass is recorded with every band rather than applied here, so that
  [sensfunc](task:sensfunc) can take the extinction out band by band. It is
  read from the header's `AIRMASS` -- run [setairmass](task:setairmass) first,
  so it is the effective airmass of the exposure and not the one at the
  shutter's opening. Without one, as in IRAF, it is worked out from `RA`,
  `DEC` and `HA` (or `ST`) at `--observatory`; failing that `--airmass` is
  used and written into the header. Only the difference in extinction
  across each band is corrected here; a spectrum already corrected
  (`EX-FLAG = 0`) has the correction taken out again.

**Every aperture.** Each aperture of each spectrum is measured, or those
`--apertures 1,3-5` names. With `--no-samestar` each aperture names its own
star (`--star-name hz44,feige34`, in aperture order). `--bandwidth 40
--bandsep 50` measures in bands of your own, their magnitudes interpolated
in the star's table.

**Beam switching.** With `--beam-switch` the spectra of an aperture come as
pairs, an object (`OFLAG = 1`) and its sky (`OFLAG = 0`) in either order; the
sky is subtracted (rebinned onto the object first if their dispersions
differ) and the file records `[object]-[sky]`. One left without a partner is
named and skipped.

**Choosing bands by hand.** `--interact` shows each spectrum with its bands:
`a` at both ends of a stretch adds a band there, `d` deletes the one nearest
in wavelength, `:show` lists them, `q` goes on. As in IRAF, a band changed
stays changed for that aperture's next spectrum, so a telluric band deleted
once is gone from the whole night. `--cursor file` reads the same keys from
a file (`wavelength y 1 key` per line), and `--answer` is IRAF's answer to
*Edit bandpasses?* (`YES!` every spectrum, `no` none).

**Blackbodies.** `--caldir blackbody --star-name V --mag 10.3 --magband V
--teff A0V` scales a blackbody to the star; a magnitude in another band is
converted through the star's colours in `params.dat`. IRAF reads those
colours two columns to the right of the band asked for (a K5V's V turned
into J by K-H, -0.11, instead of J-V, -2.13); harwood reads the band's own.

**IRAF's file.** `--format iraf` writes IRAF's own std text, which IRAF's
sensfunc reads; it is identical, line for line, to what IRAF writes for the
same spectra. harwood's sensfunc reads both, and a file keeps the format it
was started in.

**The file accumulates.** Each run appends to `std` rather than replacing
it: run `standard` on a second star, and a third, and sensfunc fits them all
together, which is how the solution covers more wavelength than any one
star does, and how stars at different airmasses separate the atmosphere from
the instrument.
