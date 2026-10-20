Finds and removes cosmic rays on single frames. Two algorithms:

- **`lacosmic`** (the default), L.A.Cosmic (van Dokkum 2001): a cosmic ray
  has an edge sharper than the optics can make, and the Laplacian of the
  image finds edges. Validated against astroscrappy, the reference
  implementation, on the same frames.
- **`iraf`**, IRAF's 1989 `cosmicrays`: a pixel standing above its
  neighbours by a flux ratio. It cannot tell a cosmic ray from the core of
  an undersampled star, because on that statistic they are the same thing,
  which is why IRAF's own help tells you to check the result by eye. It is
  here because repeating an old reduction needs the method it was done with.

```example
title: Clean the frame the cosmic rays hit
after: reduced
run: harwood cosmicrays red/obj003.fits --output cr_ --gain 2 --rdnoise 10
output: |
  obj003.fits │ 18
```

obj003 of the sample night was hit by eight cosmic rays of two pixels each.
All sixteen pixels are in the mask, with two neighbours of them the growth
step took; the cleaned frame is `cr_obj003.fits` and the mask
`cr_obj003.mask.fits`, ready for [fixpix](task:fixpix).

**The gain and read noise matter.** The method compares edges against the
expected noise, so a gain wrong by a factor of two moves every threshold.
Give `--gain` in electrons per count and `--rdnoise` in electrons. For a
long-slit spectrum, `--dispaxis` says which way the spectrum runs, so that
its smooth flank is not taken for structure.

## Checking it on your own data: `--verify`

Several exposures of one field disagree only where something happened in
one of them, which is ground truth for cosmic rays. With `--verify` and the
other exposures, harwood scores the detector on your data -- how many of
the real hits it found, how many it invented -- instead of asking you to
trust it.

That needs a scene that holds still. On the sample night the telescope
drifted between exposures, and harwood says so rather than report a
score it cannot measure:

```example
title: A verification the frames cannot support
after: reduced
run: harwood cosmicrays red/obj003.fits --verify red/obj001.fits,red/obj002.fits --gain 2 --rdnoise 10 --output v_
output: |
  the frames disagree far more on bright pixels than on faint ones
```

Verify against frames of a static field -- darks, or a set where nothing
moved -- and for a moving field reach for [imcombine](task:imcombine) with
rejection, which uses the same information without needing the scene to
hold still.
