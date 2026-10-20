Decides which arc calibrates which exposure, and writes the answer into
each spectrum's own header -- `REFSPEC1` and `REFSPEC2`, in IRAF's form --
so that the file says which arc calibrates it, and still says so next year.
[dispcor](task:dispcor) reads it from there.

```console
$ harwood refspectra 'sp00*.ms.fits' --references p/sp0018.fits --references p/sp0025.fits
│ sp0019.ms.fits │ sp0018 + sp0025 │ 0.684 0.316 │ between the arcs 56 minutes after one and 120 before the other
│ sp0024.ms.fits │ sp0018 + sp0025 │ 0.088 0.912 │ between the arcs 160 minutes after one and 15 before the other
```

(The 1992 coudé-feed night of [the recipe](page:recipes/1992-spectroscopic-night).)

`--select`, as in IRAF:

| | |
|---|---|
| `interp` (the default) | both arcs either side in time, weighted by how near each is |
| `nearest` | the one nearest in time |
| `preceding`, `following` | the one before, or after |
| `average` | the two either side, equally |
| `match` | an arc named explicitly, with `--references` |

A night has an arc at each end because a dispersion solution belongs to a
moment, not to a spectrograph: between them the flexure is smooth, so an
exposure in the middle is better served by an interpolation than by
whichever arc happens to be nearer.

The time sorted on is `UTMIDDLE`, the middle of the exposure, which
[setairmass](task:setairmass) writes; run it first. Times before
`--timewrap` (17 hours) belong to the next day, so a night that crosses
midnight sorts in the order it was observed. Without `--references`, every
solution in `database/` is a candidate. An assignment already in a header
is kept unless `--override`.
