The surface that maps a long-slit frame onto wavelength, as IRAF's
`longslit.fitcoords` fits it: the lines of the lamp that [identify](task:identify)
and [reidentify](task:reidentify) measured along the slit, every column (or line)
they traced, gathered into one fit of the wavelength as a function of both pixel
coordinates. [transform](task:transform) inverts it to straighten the frame.

```console
$ harwood fitcoords cuar_btpf_bp --no-interactive
$ harwood fitcoords arc1,arc2 --combine --fitname both --no-interactive
$ harwood fitcoords cuar_btpf_bp --xorder 4 --yorder 5 --function legendre --no-interactive
```

The features are read from `database/id<image>` -- harwood's own records or
IRAF's. A feature is used when it has a wavelength and a weight above zero; IRAF
wrote its positions in physical pixels, and they are taken to the frame's logical
ones through LTV and LTM, as IRAF does (harwood's records are logical already).
The fit is gsurfit's, in double precision: a surface of orders 2 and 2 first,
then one of `--xorder` and `--yorder` with cross terms fitted to what the first
leaves, the two added. With a single line identified the surface is the position
along the dispersion plus a function across the slit.

The fit goes to `database/fc<fitname><image>` (with `--combine`, one fit of all
the images, `database/fc<fitname>`), appended as IRAF appends it; transform reads
the last record of the name. `--deletions` (deletions.db) lists points to fit
with weight zero. The log -- the number of features, the function and orders,
the fitted wavelength at the four corners -- goes to `--logfiles`, STDOUT shown.
The second corner of the log is evaluated at (xmax, xmin), as IRAF evaluates it.

Not ported: fitting by hand in IRAF's igsfit graph (`--interactive` stops with a
note), and the graph of the fit IRAF saves in its plotfile; the course only looks
at the fit there, and `--no-interactive` gives the same surface. On the 7.2 arc (her cuar_btpf_bp, 737 features) and on 12 synthetic
cases -- both axes, legendre, low orders, LTV, combine, a fit name, deletions, one
line, a fit written twice -- the file is IRAF 2.18's to the last digit of every
coefficient, and so is the log.
