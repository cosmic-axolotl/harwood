Adds images pixel by pixel, averages them or takes their median, as IRAF's
`imutil.imsum` does, with the lowest and highest values of each pixel left
out first if asked: `--low-reject` and `--high-reject` are a number of
pixels from 1 on and a fraction of the images below 1 (0.25 of 8 images is
2). The median rejects all but the middle value -- of an even number, the
upper of the two in the middle, which is IRAF's.

```console
$ harwood imsum im010.fits,s011.fits aver1.fits --pixtype r --calctype r --option average --verbose
$ harwood imsum 'flat*.fits' flatsum.fits --low-reject 1 --high-reject 1 --hparams EXPTIME
```

Two types decide the numbers, as in IRAF. The calculation type is the
highest of the inputs' (a short and an unsigned short together make int)
unless `--calctype` names one; the output's is the calculation's unless
`--pixtype` does -- each by name or first letter, `r` for real. An integer
calculation wraps a sum too large for it and truncates an average; a real
result written as an integer image is truncated toward zero. An unsigned
short calculation has no routine of its own in IRAF, and runs in real.

With nothing rejected, IRAF adds the images fifteen at a time, keeping the
partial sum in the output image itself -- so in its pixel type: averaging
twenty frames into a short output truncates, and can wrap, the partial sum
of the first fifteen. harwood does the same. A NaN or an infinity in a real
image is read as 0, as IRAF's IEEE input reads it.

`--hparams` adds up (or, for the average, averages over every image) the
header keywords it names, with `*` and `?` patterns, ten at most -- not for
the median. The value is read and written as IRAF reads and writes it, to
the last digit IRAF gets wrong.

```example
title: Add three exposures and their exposure times
after: reduced
run:
  - harwood imsum 'red/obj00*.fits' sum.fits --hparams EXPTIME
  - harwood hselect sum.fits '$I,EXPTIME'
output: |
  -> sum.fits
  sum.fits	180.0
```

The frames carry their variance, and imsum leaves it behind and says so:
IRAF's imsum keeps no uncertainty. For a stack whose noise will be needed,
[imcombine](task:imcombine) adds the variance planes too.

Every case compared -- one to twenty images of each pixel type, every option,
rejection by number and by fraction, each pair of calculation and output
types, sections, 1-D images and cubes -- gives IRAF 2.18's file pixel for
pixel, its BITPIX and its header parameters; what IRAF refuses (images of
different sizes, a rejection that leaves nothing, a type it does not know)
harwood refuses in the same words, with what to do.
