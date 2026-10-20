Combines bias frames into one master bias. It is
[imcombine](task:imcombine) with IRAF's settings for bias frames, from
ccdred's `zerocombine.cl`: average, after dropping the highest value at each
pixel (`reject=minmax`, `nlow=0`, `nhigh=1`), of the frames of type `zero`.

Dropping the highest value is crude and predictable, which is what a
handful of bias frames needs: there is no noise model worth trusting in
three frames, and nothing to protect but the odd cosmic ray. Process the
frames first -- overscan and trim -- so the master has the size of the
frames it will be subtracted from.

```example
title: The master bias
run:
  - harwood ccdproc @bias.lis --overscan --trim --output bias
  - harwood zerocombine bias/bias00*.fits Zero.fits
output: |
  rejected: 10000
  -> Zero.fits
```

Ten thousand values rejected is one per pixel of the 100 × 100 frames: the
highest of the three, everywhere. The preset's values are a separate origin,
so `harwood lpar zerocombine` says *preset: zerocombine* beside them rather
than passing them off as defaults; anything you give overrides them.
