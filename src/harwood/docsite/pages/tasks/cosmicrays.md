Finds and removes cosmic rays on single frames. Two algorithms:

- **`lacosmic`** (the default), L.A.Cosmic (van Dokkum 2001): a cosmic ray
  has an edge sharper than the optics can make, and the Laplacian of the
  image finds edges. Validated against astroscrappy, the reference
  implementation, on the same frames.
- **`iraf`**, IRAF's 1989 `cosmicrays`: a pixel standing above its
  neighbours by a flux ratio. It cannot tell a cosmic ray from the core of
  an undersampled star, because on that statistic they are the same thing,
  which is why IRAF's own help tells you to check the result by eye. It is
  here because repeating an old reduction needs the method it was done with:
  see [IRAF's algorithm](#irafs-algorithm).

```example
title: Clean the frame the cosmic rays hit
after: reduced
run: harwood cosmicrays red/obj003.fits --output cr_ --gain 2 --rdnoise 10
output: |
  obj003.fits │ 18
```

obj003 of the sample night was hit by eight cosmic rays of two pixels each.
All sixteen pixels are found, with two neighbours of them the growth step
took. The cleaned frame is `cr_obj003.fits`, and the eighteen pixels are
flagged in its DQ -- bit 8, *cosmic ray* -- so they stay known for what they
are after the replacement: [imcombine](task:imcombine) leaves them out and
the viewer can show them. Their variance becomes their neighbours'.

`--no-clean` only flags them and leaves the data as it was, which is what a
combination wants -- it has the other frames to fill the gap with -- and
what [fixpix](task:fixpix) takes with `DQ` as its mask. `--crmask` also
writes them as a mask image of their own, and `--format iraf` writes the
frame as IRAF reads it, with the DQ as the mask its `BPM` names.

**The gain and read noise matter.** The method compares edges against the
expected noise, so a gain wrong by a factor of two moves every threshold.
Left out, they come from the frame's `GAIN` and `RDNOISE`; with neither,
1 and 0 are used and the run says so.

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

## IRAF's algorithm

`--algorithm iraf` gives the pixels, the mask and the `CRCOR` card IRAF 2.18
gives, on real and integer images. A pixel is a candidate when it is the
highest of a window of `--window` 5 or 7 and stands `--threshold` above its
mean; the replacement is the mean of its four nearest, leaving out the next
highest. A plane fitted to the border comes off, and `--fluxratio` tells the
two kinds apart: 100 times the mean left, over the peak. Stars have a high
ratio and cosmic rays a low one. `--npasses` looks again around what was
replaced.

Two of IRAF's habits come with it, because an old reduction has them. In each
pass after the second, the last pixel the previous pass replaced still reads
as it was: IRAF's output buffer has not written it yet. So a pixel can be
found twice, and `removed=` in `CRCOR` counts it twice, and it can also change
what is found next to it. And its 7 by 7 window takes one point of the
border from the row inside it, `z[16] = f[i4]`, and never looks at the one
it should.

**Telling stars from cosmic rays.** `--interactive` opens the flux against
the flux ratio of the candidates of each frame, with IRAF's keys: `t` sets
the threshold at the cursor, `d` and `u` decide one candidate, `e` and `v`
those in a region, `a` shows only the training objects, `s` the pixels around
a candidate, space prints its column and line, `q` goes on. The mouse does
what `w` does there. A threshold you set goes on to the next frame.

`--train` lets the objects you know set it: in the first frame's image, `s`
at a star and `c` at a cosmic ray, each the candidate within three pixels,
and `g` for the graph, with the threshold that errs least on them. Those stay
as you said, whatever the threshold. `--savefile` adds the marks to a file
`--objects` reads back; `--objects` and `--cursor` are files of `x y wcs key`
instead of the window. What is at or under zero flux has no place on the log
axis, and the keys cannot reach it. IRAF's `plotfile` is not here: the graph
is the panel.
