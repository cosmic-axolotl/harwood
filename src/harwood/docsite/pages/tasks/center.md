Recentres a list of stars, as IRAF's `center`: from each starting position
it finds the centre of the light in a box `--cbox` wide, by IRAF's centring
algorithms -- the marginal centroid (the default), a Gaussian fit
(`gauss`), or the optimal filter (`ofilter`) -- iterating until the centre
settles.

```example
title: Better centres for the stars daofind found
after: found
run: harwood center red/obj001.fits obj001.coo obj001.ctr --cbox 5
output: |
  flagged: 0
```

The table keeps where each star started (`xinit`, `yinit`), where it ended
(`x`, `y`), the shift and its error, and IRAF's code for anything flagged
(`cier`): too faint (`--minsnratio`), moved further than `--maxshift`, too
near the edge. Its `x` and `y` are the new centres, so it can be handed
straight to [phot](task:phot) with `--calgorithm none`.

[phot](task:phot) centres every star itself before measuring it, with the
same parameters, so `center` on its own is for when the centres are what
you want -- to measure a shift, or check a list from another frame.

`--no-emission` (IRAF's `emission = no`) centres on dips rather than
peaks -- absorption features, or a star in negative: the box is turned
over about its top before the centre is found, as apfitcen does; checked
against IRAF for every algorithm.


## With the cursor

`--interactive` hands the window's keys to the task, as IRAF's
`interactive+` does, and `--icommands file` reads them from a file of
`x y 1 key` lines instead -- IRAF's icommands, `\040` for the space bar --
which is how they were checked: on IRAF 2.18's own runs, the same rows,
numbered the same way.

| key | |
|---|---|
| space | measure at the cursor and write the row |
| `f` | measure at the cursor without writing |
| `c` | centre at the cursor |
| `t` / `s` | the sky at the cursor / around the current centre (phot) |
| `a` | the sky averaged over several positions: `t` at each, `q` to finish (phot) |
| `p` / `o` | magnitudes with the sky as it is; `o` writes them (phot) |
| `m` / `n` | move to / measure the next star of `--coords` |
| `l` | measure every star left in the list |
| `r` | back to the start of the list |
| `e` | the last error messages |
| `q` | done |

`:m 5` and `:n 5` move to or measure star 5; `:annulus 12`, `:cbox 7`,
`:apertures 3,5` and the other parameters are set by name for what follows.
A cursor within a FWHM of the star just fetched from the list measures that
star, as in IRAF. Each row carries `id`, its place in the output, and `lid`,
its place in the list -- 0 for one the cursor chose.
