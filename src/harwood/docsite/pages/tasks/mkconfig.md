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

**Typed, as IRAF asks.** Any section given as `STDIN` is typed at IRAF's
prompts instead: the catalog's or the observations' columns one at a time, a
name and a column number each; the equations one at a time, each a label, the
equation (a line ending in `\` goes on to the next), its parameters to fit and
those held constant, a name and an initial value each. Ctrl-D ends each list
-- the columns, the parameters, the equations -- and `?` shows IRAF's help for
the section. `--verify` asks again for each entry, to change it. In the window
each prompt is a question, and Cancel is Ctrl-D; `--answers` reads the lines
from a file, where a line `<EOF>` stands for Ctrl-D.

```console
$ harwood mkconfig bv.cfg landolt obsA STDIN
```

IRAF's own slips are kept, since a file made with it is what is being matched:
a label verified to another is cut to the new one's length rather than
replaced; a parameter renamed at verify keeps its old value; the parameters to
fit, and those held constant, have room for 160 characters in all, past which
the line is cut and no more equations are asked for; a line of the file longer
than 78 characters is broken at the 78th, wherever that falls -- which the
check then reports. Type long equations as a template file instead.

**Then edit it.** A template's equations name every filter of its system
and start from generic coefficients; with only B and V observed, the
Landolt template's U, R and I equations name variables the observations
do not have, and the check says so, equation by equation. Delete those,
and set the initial values near what the night gave. `--edit` opens the new
file in the editor before it is checked, as IRAF does -- a dialog in the
window, `$EDITOR` at a terminal; IRAF has it on, harwood off, so that a run
can be repeated.

**Identical to IRAF.** The file is IRAF's mkconfig's, byte for byte, for
the standards directory's catalogs and for one's own format files and
templates, and typed: what is asked and said, and the file, on eight
sessions -- each section typed, with verify and without, the help, every
error message, and the 160-character limit.
