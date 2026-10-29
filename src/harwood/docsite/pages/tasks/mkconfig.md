Makes a photcal configuration file from its three sections -- the
catalog's columns, the observations' columns, and the transformation
equations -- and checks it as [chkconfig](task:chkconfig) does.

```console
$ harwood mkconfig bv.cfg landolt observations landolt
```

**Where the sections come from.** The catalog section from a format file,
`f<catalog>.dat` here or in the standards directory, or the file named;
the observation section from `f<observations>.dat`, which
[obsfile](task:obsfile) writes next to the observations; the equations from
a template, `t<transform>.dat` here or in the standards directory, or the
file named. They are copied one after the other, as IRAF copies them.
`--template` copies a whole configuration instead.

**The standards directory.** `--catdir` is `photcal$catalogs/`, which is
IRAF's own directory of standards catalogs, shipped with harwood: Landolt
(1983, `landolt`; 1992, `nlandolt`), Elias, Odewahn, Porter, UKIRT, NICMOS
and ARNICA, each with its format file and a template of equations. A
catalog of one's own made with [mkcatalog](task:mkcatalog) comes with its
format file too.

**Then edit it.** A template's equations name every filter of its system
and start from generic coefficients; with only B and V observed, the
Landolt template's U, R and I equations name variables the observations
do not have, and the check says so, equation by equation. Delete those,
and set the initial values near what the night gave. The prompting IRAF
does when a section has no file, and its editor session, are not ported:
the file is plain text.

**Identical to IRAF.** The file is IRAF's mkconfig's, byte for byte, for
the standards directory's catalogs and for one's own format files and
templates.
