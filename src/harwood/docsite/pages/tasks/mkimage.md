Makes an image of simple values, or edits one with them -- IRAF's `mkimage`, from
ccdred's test package: a sloped plane and Gaussian noise. A test frame whose every
value is known.

```example
title: A frame of 100 with a gentle slope and some noise
run: harwood mkimage ramp make 100 --dims '40 30' --slope 0.2 --sigma 3 --seed 1
output: |
  -> ramp.fits
```

The options are `make` (a new image of `--ndim` dimensions, `--dims` long, of
`--pixtype`), and `replace`, `add` and `multiply` (an image there is, or a section of
it: `ramp[1:20,*]`). The plane's value is its mean, at the middle of the image (or of
the section), and it rises by `--slope` a pixel along both the lines and the columns.

The noise is IRAF's, and only nearly random, as IRAF's is: a row of twice the line's
length of Gaussian numbers is made at each call, and each line takes its pixels from a
place chosen at random in it. The generator's seed carries from one call to the next, as
in one IRAF process; `--seed` starts it again from a number.

What IMIO does is done here too, and changes the numbers: a section one pixel wide is a
line, not a column of one-pixel lines (`ramp[15:15,*]`), and a new one-dimensional image
has no second length, which moves its plane by half a step. The images are IRAF 2.18's,
bit for bit.
