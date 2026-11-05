Lists the apertures of spectra, one line each, as IRAF's slist prints them.

```console
$ harwood slist 'night1/*.w.fits'
sp0019.w 1 1 1 0 4013.257 2.071906 1024 DHCep 6707
sp0019.w 2 2 2 0 4011.886 2.072128 1024 DHCep 6707
$ harwood slist sp0019.w --long-header
sp0019.w: DHCep 6707
    EXPTIME = 600.00   UT = 3:12:45.5      ST = 1:40:12.3
    RA = 22:07:40.20   DEC = 57:43:36.0    HA = -0:58:10.20    AIRMASS = 1.084
    1 1 1 0 4013.257 2.071906 1024 DHCep 6707
```

After the name come the aperture's line in the file, its aperture and beam
numbers, its dispersion type (IRAF's DC-FLAG: -1 still in pixels, 0 linear, 1
linear in log10, 2 anything else, read off the wavelength column), the first
wavelength and the mean step to the last, the number of pixels, and the title:
the aperture's own, else the spectrum's. The first wavelength and the step are
computed in real, as IRAF's shdr computes them, so a step of 1.6 can say
1.599999. `--long-header` puts the exposure time, UT, sidereal time, RA, DEC,
hour angle and airmass of each spectrum first; `--apertures 1,3` lists some.

Four sessions IRAF 2.18 ran -- linear, logarithmic and pixel apertures, a beam
not its aperture's, titles and none, the long header, some apertures, and an
image that does not exist -- print the same, line for line. The beam is the
aperture's own number unless the spectrum says otherwise (`APBEAM` in its
header), as apall writes it.
