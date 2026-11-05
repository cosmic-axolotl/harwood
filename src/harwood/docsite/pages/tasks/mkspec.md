Makes an artificial spectrum: flat, a ramp, or a blackbody, of so many pixels
and, with more than one line, a frame of the same spectrum on every line.

```console
$ harwood mkspec flat 50 --image-title flat --function 1 --constant 12.5
$ harwood mkspec bb 200 --function 3 --start-wave 3000 --end-wave 9000 --temperature 9500
$ harwood mkspec frame 120 --nlines 40 --function 3 --temperature 3300
```

`--function 1` is flat at `--constant`; `2` a ramp from `--start-level` to
`--end-level`; `3` a blackbody of `--temperature` from `--start-wave` to
`--end-wave` Angstroms, divided by the peak of Wien's law, 1.2865e-4 T⁵, so that
it reaches 1 where its peak is in the range. The spectrum has no wavelengths,
as IRAF's has none: [dispcor](task:dispcor) or [rspectext](task:rspectext) give
it some. One line is a spectrum of one aperture; more, a frame for apall.

The pixels are computed as IRAF computes them, in real -- the powers by
squaring, as f2c's `pow_ri` does, the exponential in double, the wavelength
stepped along with its error carried on -- and four sessions IRAF 2.18 ran, a
flat, a ramp and two blackbodies, one of them four lines, are IRAF's to the
bit.
