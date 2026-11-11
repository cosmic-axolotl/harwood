IRAF's `onedspec.splot`: a spectrum plotted and measured with the cursor's keys.
The spectrum is opened as onedspec opens it -- a 1-D image with its world system
(an IRAF onedspec, apall's `--format onedspec`), a line of a long slit summed over
nsum along DISPAXIS, an aperture of an IRAF multispec image with its linear,
log-linear or nonlinear dispersion, or one of harwood's own spectrum files.

```console
$ harwood splot ngc660_bg_sub.0002.fits
$ harwood splot cuar_transform.fits --line 129 --nsum 10
$ harwood splot l0025.ms.fits --line 1 --gcommands keys.txt --answers answers.txt
```

The keys are IRAF's, pressed with the cursor over the graph in splot's window
(they are listed below it), or read from a cursor file (`--gcommands`), a line
each, `x y 1 key` in the graph's coordinates -- `\040` is the space bar -- or
`:command`. A question (the box of `s`, the constant of `f`'s `m` and `p`, the
next image of `g`, a line list's name) is asked on the window's command line, or
read from `--answers`, a line each. What splot prints shows on the window's
status line; `:.snap file.png` saves the graph.

* space -- the cursor's x and y, and the spectrum's value at x;
* `m` `m` -- mean, rms and signal to noise between two points;
* `e` `e` -- the equivalent width, flux and centre against a straight
  continuum between two points;
* `k` and a second key -- a gaussian (`k`/`g`), lorentzian (`l`) or voigt (`v`)
  fitted between two continuum points, with Monte-Carlo errors from
  `--nerrsample`, `--sigma0` and `--invgain`;
* `d` `d` -- several lines deblended: mark each with `g`, `l` or `v` (or read
  them with `f` from a file, or type one with `t`), `q`; then the fit's choices,
  positions and widths fixed, single or all, the background or not; the
  overplot; `+` and `-` through the lines' results, `r` the rms, `q`;
* `-` `-` -- the last fit subtracted from the spectrum between two points;
* `h` and `a`, `b`, `c` (width at half the line) or `l`, `r`, `k` (width at
  the cursor's level, continuum 1);
* `s` (box smoothing), `f` (log, exp, sqrt, abs, inverse, +k, *k, and the
  arithmetic of two spectra), `j` and `x` (pixels fixed by hand), `l` and `n`
  (flux per wavelength and per frequency), `$` (pixels and back), `v` (velocity
  about the cursor and back), `:units`, `g` `#` `(` `)` (another image, line or
  aperture);
* `t` -- a continuum fitted with icfit (its window, or the keys that follow up
  to its `q`) and the spectrum divided by it (`/`), less it (`-`), replaced by
  it (`f`), cleaned of what it rejects (`c`), or left as it is (`n`);
* `p` -- new linear coordinates from a start, an end or an interval; `u` and
  then `d`, `z` or `l` -- a doppler shift, a zero point or a linear dispersion
  from features marked and named;
* `i` -- the spectrum written, a new image or over its own (asked first);
* `y` -- a standard star's calibration fluxes boxed over the spectrum (the star
  asked for, from `--caldir`; an unknown one lists the directory's stars);
* `:#` a comment in the log, `:log`, `:nolog`, `:show`, `:label`, `:mabove`,
  `:mbelow` (a label, or a tick and a label above or below the spectrum; `%.1f`
  writes the cursor's x), `:nsum` and `:dispaxis` (a long slit summed or read
  along the other axis), and the options.

The measurements go to the terminal and to `--save-file` (splot.log), with the
date and the spectrum's name when a measurement is the first of a spectrum and
the column header when the key changes, as IRAF writes it.

Not ported: the help pages of `?` and `/` (the keys are listed under the graph).
On 38 cursor files over an onedspec, a long slit with nsum 1 and 5 and a
multispec with a chebyshev dispersion, what splot prints and what it logs are
IRAF 2.18's character for character, each graph's window, title and label are
those of IRAF's metacode, and the spectra `i` writes are IRAF's pixel for pixel
with its dispersion (a log-linear multispec one to the 14th figure).
