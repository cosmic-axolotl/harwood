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
  shutter's opening.

**The file accumulates.** Each run appends to `std` rather than replacing
it: run `standard` on a second star, and a third, and sensfunc fits them all
together, which is how the solution covers more wavelength than any one
star does, and how stars at different airmasses separate the atmosphere from
the instrument.
