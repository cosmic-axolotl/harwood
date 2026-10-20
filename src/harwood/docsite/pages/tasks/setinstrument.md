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

As in IRAF, the instrument's settings come with it: IRAF's setinstrument
runs `ccddb$kpno/direct.cl` after choosing `direct.dat`, and harwood reads
the same file -- `default.cl`'s ccdproc and combine settings, then
direct's own `flatcor = yes` -- and puts each one harwood has into the
project file's `params`, where `harwood lpar ccdproc` shows them and any of
them can be changed. `--site` is the ccddb directory a bare name is looked
for in (`kpno`, as in IRAF). The table lists every setting and whether it
was applied: ccdred's package settings (the logfile, the plot file, backups)
are harwood's own business, and a setting naming a header keyword where
harwood wants a number (`combine.gain = "gain"`) is left out, said so.

The choice goes into the reduction's `harwood.yaml`, not a setting on the
machine, so the reduction carries its own conventions and someone else can
repeat it. Without one, harwood reads the common keywords under their own
names -- `IMAGETYP`, `FILTERS`, `EXPTIME`, `BIASSEC`, `TRIMSEC` -- and calls
anything it cannot classify `unknown`, rather than guessing.
