The surface that maps a long-slit frame onto wavelength, as IRAF's
`longslit.fitcoords` fits it: the lines of the lamp that [identify](task:identify)
and [reidentify](task:reidentify) measured along the slit, every column (or line)
they traced, gathered into one fit of the wavelength as a function of both pixel
coordinates. [transform](task:transform) inverts it to straighten the frame.

```console
$ harwood fitcoords cuar_btpf_bp --interactive
$ harwood fitcoords cuar_btpf_bp --cursor keys.cur --answers answers.txt
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

By hand (`--interactive`) the fit opens in IRAF's igsfit graph: at first the
position along the dispersion against the residual, each feature coordinate a +,
a deleted one an x. The keys are IRAF's:

| key | what it does |
| --- | --- |
| `c` | print x, y, the wavelength, the weight, the surface and the residual of the nearest point |
| `d` | delete the nearest point, then `p` it alone, or `x`, `y`, `z` every point of its column, line or feature |
| `u` | undelete the same way (a point of the deletion list comes back with weight 1) |
| `f` | fit again, with the function and orders the colon commands set |
| `x` `y` | choose the graph's axes: `x` `y` `z` (wavelength) `s` (surface) `r` (residual); drawn at the next `r` |
| `z` | zoom on the points sharing the nearest one's `x`, `y` or `z`; `n` and `l` go to the next and last set, `p` back to all |
| `r` | redraw; `w` gtools' window keys; `?` help; `q` quit |
| `:show` `:corners` | the orders and the fitted wavelength at the four corners |
| `:function` `:xorder` `:yorder` | set them (orders of at least 2); `f` fits with them |

On `q` the log is written and the window asks "Write coordinate map to the
database?"; the deleted points go to `--deletions` either way, for the next run.
As in IRAF, `x` and `y` only choose the axes until `r`, the residuals' rms in the
title counts the deleted points too, and the orders set by `:xorder` are the next
image's as well. One difference: a fit that cannot be made (no degrees of freedom
left) ends IRAF's task; the window says so and keeps the fit before it.

Without a window, `--cursor` reads the keys from a file, IRAF's cursor format
(`x y 1 key`, or a bare `:command`), on from one fit to the next, and
`--answers` the replies to "Fit <image> interactively" (no, yes, NO, YES) and
"Write coordinate map to the database" (yes, no), a line each. `--plotfile`
writes the graph as each fit leaves it, as SVG.

On the 7.2 arc (her cuar_btpf_bp, 737 features) and on 12 synthetic
cases -- both axes, legendre, low orders, LTV, combine, a fit name, deletions, one
line, a fit written twice -- the file is IRAF 2.18's to the last digit of every
coefficient, and so is the log. By hand, 20 sessions of keys against IRAF's own
(colon commands, every kind of delete and undelete, the axes, the three zooms,
the window and :/ commands, transpose, a log axis, the bells, two images in a
row, one line) and one on the 7.2 arc print the same text to the byte, write the
same fit, deletion list and log, and draw every graph with the same window and
each mark where IRAF's metacode has it.
