IRAF's `mkobjects`: stars and galaxies from a list, in an image or making one. The
answer is known, so it is how a reduction is checked. For a seed the numbers are IRAF's.

A line of the list has `x y magnitude`, a star; and a galaxy goes on with its model, radius,
axial ratio and position angle:

```
100.3 120.7 15.0
200.5 80.2 14.0 expdisk 12.0 0.6 30.0
310.1 250.9 13.5 devauc 8.0 0.8 110.0
```

The models are `expdisk`, `devauc`, `sersic` with its index, as `sersic2.5`, an image, whose
half width is the radius, or a text file with the profile, a value a line from the centre to
the radius; if its first value is 0 the file is the integral of the profile. A galaxy is convolved with the star's seeing. Off the image is passed over, what does not begin with
three numbers too.

`--star` is `moffat`, `gaussian`, or such an image or text file, of half width `--radius`, with `--beta`,
`--ar` and `--pa`. A magnitude is `--magzero` for one data number in a second, for `--exptime`
seconds. `--distance` divides positions and radii and dims. `--background`, per second,
and the noise are for a new image, the noise on all of it; on an image that exists, on the
objects only.

These are the bits of IRAF 2.18, seeing and all. `save` in the list only saves time, and is
not needed here; an object of no known model is said and passed over.
