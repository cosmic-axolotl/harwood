IRAF's `mknoise`: background, read noise, Poisson noise and cosmic rays, added to an
image or making one. For a seed the numbers are IRAF's own, the generator included.

```example
title: A blank frame with the noise of a night
run: harwood mknoise blank.fits --ncols 200 --nlines 200 --background 500 --gain 2 --rdnoise 8 --poisson
output: |
  -> blank.fits
```

`--cosrays` names a file of `x y energy`, in electrons. If it does not exist and `--ncosrays`
says how many, that many are made at random, up to `--energy`, and written to it. Their shape
is a gaussian of `--radius`, `--ar` and `--pa`. Noise goes on after them, line by line:
the Poisson first, then the read noise.

Without `--output` an image that exists is changed. A new one takes its header from
`--header`, IRAF's standard one by default, and `GAIN` and `RDNOISE` go in either way.
`--no-comments` leaves out the record of the parameters.

The numbers are the bits of IRAF 2.18. IRAF's package parameter `ranbuf`, which reuses
values to go faster at the cost of what is random, is not here.
