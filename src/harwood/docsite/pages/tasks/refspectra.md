Decides which arc calibrates which exposure, and writes the answer into
each spectrum's own header -- `REFSPEC1` and `REFSPEC2`, in IRAF's form --
so that the file says which arc calibrates it, and still says so next year.
[dispcor](task:dispcor) reads it from there. It is IRAF's refspectra,
ported from `noao/onedspec/dispcor/ref*.x` and checked against IRAF 2.18 on
fifteen runs: the same assignments, the same weights, the same lines.

```console
$ harwood refspectra 'sp00*.ms.fits' --references p/sp0018.fits --references p/sp0025.fits
[sp0019.ms] refspec1='sp0018 0.68366009'
[sp0019.ms] refspec2='sp0025 0.31633991'
...
[sp0024.ms] refspec1='sp0018 0.087728649'
[sp0024.ms] refspec2='sp0025 0.91227132'
```

(The 1992 coudé-feed night of [the recipe](page:recipes/1992-spectroscopic-night).)

## Choosing

`--select`, as in IRAF:

| | |
|---|---|
| `interp` (the default) | the arcs either side, weighted by how near each is; outside them, the nearest |
| `nearest` | the one nearest |
| `preceding` | the nearest one before -- and, as in IRAF's code, nothing when there is none (IRAF's manual says the following one; harwood says so when it happens) |
| `following` | the nearest one after, or before when there is none after |
| `average` | the first two references named, equally, whatever their time |
| `match` | each input with the reference in the same place of the list |

"Near" is measured on `--sort`, by default `JD`, the Julian date of
mid-exposure [setjd](task:setjd) writes; with `--time` the keyword is an
hour of the day (`UT`, `UTMIDDLE`), wrapped at `--timewrap` so a night that
crosses midnight sorts in the order it was observed. A reference must share
the input's `--group` value -- by default `LJD`, the night -- so an arc
from another night is never used; `none` groups nothing. When a header has
no JD or LJD, they are worked out as setjd would, and a note says so.

A night has an arc at each end because a dispersion solution belongs to a
moment, not to a spectrograph: between them the flexure is smooth, so an
exposure in the middle is better served by an interpolation than by
whichever arc happens to be nearer.

## Apertures and references

`--apertures` and `--refaps` choose apertures (`BEAM-NUM`) as range lists,
`1,3-5`. With `--no-ignoreaps` each spectrum takes only arcs of its own
aperture. A reference is, as in IRAF, a spectrum whose `REFSPEC1` names
itself -- or here one with a solution in `database/`; arcs named with
`--references` that are neither are used as named. Without `--references`,
every solution in `database/` is a candidate. A single `--references` that
is a text file is IRAF's reference table: lines of `inputs references`
(lists or templates), a line without references taking the previous
line's.

## Writing

The lines above are IRAF's log (`--logfiles`, `STDOUT` and any files); the
weights are single precision, printed with eight digits, and a single arc is
written without one. An assignment already in a header is kept unless
`--override`. `--confirm` asks before each one, as IRAF does by default:
`no` skips it, `yes` takes it, `YES` takes it and all the rest. `--verbose`
also logs why a spectrum was passed over. When nothing can be assigned,
IRAF's explanation -- which arcs were of another night, or another aperture
-- comes as a warning.
