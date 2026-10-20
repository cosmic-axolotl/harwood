Chooses the header translation a reduction uses: which keyword holds the
filter, the exposure, the sections, and what each value of `IMAGETYP` means.
IRAF's `ccddb` files are included -- `kpno/direct`, `ctio/cfccd_f1` and
the rest -- and a file of your own works the same way.

```example
title: The translations there are
run: harwood setinstrument --list-available
output: |
  kpno
```

```example
title: Use KPNO's direct-imaging translation for this night
run: harwood setinstrument kpno/direct
```

The choice goes into the reduction's `harwood.yaml`, not a setting on the
machine, so the reduction carries its own conventions and someone else can
repeat it. Without one, harwood reads the common keywords under their own
names -- `IMAGETYP`, `FILTERS`, `EXPTIME`, `BIASSEC`, `TRIMSEC` -- and calls
anything it cannot classify `unknown`, rather than guessing.
