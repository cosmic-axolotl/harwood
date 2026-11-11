A long-slit frame resampled onto straight lines of constant wavelength, as IRAF's
`longslit.transform` does it: the surfaces [fitcoords](task:fitcoords) wrote are
inverted, and the frame interpolated where each output pixel falls.

```console
$ harwood transform cuar_btpf_bp.fits cuar_transform.fits cuar_btpf_bp --dispaxis 2
$ harwood transform ngc660_master.fits ngc660_transform.fits cuar_btpf_bp --dispaxis 2
$ harwood transform gal.fits gal_log.fits arc --ylog --dy 0.0001
$ harwood transform rectified.fits resampled.fits '' --y1 6300 --y2 6640 --dy 1
```

The fits are read from `database/fc<name>`; one of the wavelength, and one along
the slit if there is one (an axis with none keeps its pixels). The inversion is
done on every tenth output pixel by Newton's method, walked back and forth so
that each starts from the last, and interpolated bilinearly in between; the
input is then interpolated with `--interptype` (nearest, linear, poly3, poly5,
spline3) from a buffer of its lines. With `--flux` (the default) each pixel is
multiplied by the input area the output pixel covers, so that counts are kept.

The output's limits and intervals are `--x1 --x2 --dx --nx` and `--y1 --y2 --dy
--ny`, each taken from the fits' range and the input's size when left out; an
interval given sets the number of pixels. `--xlog`/`--ylog` make the intervals
logarithmic. Off the input, pixels are the edge's, extended, or `--blank`.
`--minput` and `--moutput` carry a mask through (the output mask is a FITS file
with the mask in a PLIO extension, as IRAF writes it, named in BPM).

The output has a linear world system with the fit's units, and for onedspec
DCLOG1, DISPAXIS and DC-FLAG. The dispersion axis is the DISPAXIS keyword or
`--dispaxis` -- IRAF asks for it ("Dispersion axis ... (2)") when the frame has
none, as the course's frames do. With `''` as the fits, each image is
resampled by its own linear world system instead.

On the 7.2 frames (her cuar_btpf_bp and ngc660_master, with the fit above) and on
28 synthetic cases -- every interpolator, flux on and off, blank and edges,
log intervals, given sizes, intervals and limits, two fits averaged, several
images, integers, masks, a frame transposed or trimmed, and the image's own
world system -- the output is IRAF 2.18's pixel for pixel, with its header and
log.
