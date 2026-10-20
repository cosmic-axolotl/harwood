Works out the airmass for one position, time and site -- from a frame's
header, or from what you give it -- at the start, middle and end of the
exposure, with refraction.

```example
title: The airmass of a frame, from its header
run: harwood airmass obj001.fits
output: |
  airmass: 1.0343
```

`--ra`, `--dec`, `--date`, `--ut` and `--exptime` override the header, and
`--observatory` (or `--latitude`, `--longitude` positive west, `--altitude`)
the site. To write the result into many headers, use
[setairmass](task:setairmass).
