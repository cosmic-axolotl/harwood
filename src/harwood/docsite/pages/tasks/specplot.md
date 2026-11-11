IRAF's `onedspec.specplot`: several spectra drawn together, each above the one
before. Every aperture (and band) of every image named is a spectrum, opened as
onedspec opens it -- an IRAF onedspec, a line of a long slit summed over nsum, an
aperture of a multispec image, one of harwood's spectrum files.

```console
$ harwood specplot l0025.ms.fits
$ harwood specplot ngc660_bg_sub.0002.fits,ngc660_bg_sub.0003.fits --labels imname
$ harwood specplot l0025.ms.fits --apertures 1-3 --gcommands keys.txt --logfile sp.log
```

The automatic layout (`--autolayout`, on by default) scales each spectrum to the
first one's mean -- or offsets it to it, with `--no-autoscale` or when a mean is
not positive -- and then steps each up by the largest separation that clears one
spectrum from the next, times `--fraction`. Without it, `--scale` and `--offset`
(a value, `@file` with a line a spectrum, or a header keyword) and `--step` place
them. `--transform log` draws the logarithm of the flux.

The keys are IRAF's, pressed with the cursor over the graph in specplot's window
(they are listed below it), or read from a cursor file (`--gcommands`), a line
each, `x y 1 key` in the graph's coordinates, or `:command`:

* `a` and `i` -- a spectrum added after or before the one nearest the cursor
  (its name asked); `d` deletes the nearest, `e` brings back the last deleted;
* `s` -- the nearest spectrum shifted with the cursor: `s` again moves it, `t`
  moves it in x only, `x` and `y` in one axis, `z` scales it, `r` redraws, `q`;
* `t` -- two points marked and their coordinates typed: a new scale for the
  nearest; `u` -- one point and its coordinate: a shift;
* `v` -- velocities about the cursor's coordinate;
* `x` (no layout), `y` (offsets to a common mean), `z` (scales to it);
* `o` -- the indices renumbered with no gaps; `f` -- pixels or world
  coordinates; `l` and `p` -- a label written at the cursor, or moved there;
* `w` -- gtools' window keys; `r` redraws, `q` ends;
* `:show` and `:vshow` -- the table of the spectra, with velocities in vshow;
  `:step`, `:fraction`, `:move`, `:shift`, `:w0`, `:wpc`, `:velocity`,
  `:redshift`, `:offset`, `:scale`, `:xlpos`, `:ylpos`, `:labels`, `:ulabel`,
  `:ptype`, `:units`, `:color` -- each with `[n]` for spectrum n, or the
  nearest to the cursor; with no value, the value is shown.

A question (the spectrum `a` adds, the coordinate of `t` and `u`, a label) is
asked on the window's command line, or read from `--answers`, a line each. What
specplot prints shows on the window's status line; `:.snap file.png` saves the
graph. `--logfile` gets the table of `:vshow` at the end, appended, as IRAF
writes it.

Not ported: the help page of `?` (the keys are listed under the graph). On 19
cursor files over a multispec of three dispersions, an onedspec and a long slit,
what specplot prints, its logfile and each graph's window, title and labels are
IRAF 2.18's (metacode).
