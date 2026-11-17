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
for in (`kpno`, as in IRAF). The table lists every setting and how it was
applied. ccdred's package settings -- the logfile, verbose, the pixel type, the
plot file, backups, the subsets file, graphics and cursor -- go into `params.ccdred`,
which ccdproc, the combines, ccdlist and ccdgroups read as IRAF's tasks read the
package; an empty file name is none. The demonstration's (`demo`) also sets the
artificial frames' tasks of the [ccdtest](task:artobs) package.

The translation files are read as IRAF's hdrmap reads them: any parameter, a default
for it in a third column (`biassec biassec []`), and the processing flags too -- the old
camera format writes `fixpix bp-flag 0`, a step being done when the keyword says other
than its default.

The choice goes into the reduction's `harwood.yaml`, not a setting on the
machine, so the reduction carries its own conventions and someone else can
repeat it. Without one, harwood reads the common keywords under their own
names -- `IMAGETYP`, `FILTERS`, `EXPTIME`, `BIASSEC`, `TRIMSEC` -- and calls
anything it cannot classify `unknown`, rather than guessing.
