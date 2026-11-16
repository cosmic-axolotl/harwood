Divides the telluric bands out of spectra with the spectrum of a star that has nothing
else in it, shifted and raised to the ratio of the airmasses -- the shift and the power
found for each spectrum by cross-correlation and the least RMS.

```console
$ harwood telluric s c t
ok telluric in 1.13s
TELLURIC:
  Output: c - spectrum s
  Input: s[*,1] - star one
  Calibration: t[*,1] - tell one
  Tweak: shift = 0.70, scale = 0.988, normalization = 0.9737
  WARNING: 1 pixels outside of calibration limits
  Input: s[*,2] - star two
  Calibration: t[*,2] - tell two
  Tweak: shift = 0.70, scale = 1.012, normalization = 0.9735
  WARNING: 1 pixels outside of calibration limits
  -> c.fits
$ harwood telluric @night @night hd84937 --sample 6860:6960,7150:7350
```

Each spectrum is divided by the calibration -- the aperture of the same number, or
the first with `--ignoreaps` or when there is none such -- interpolated by `--interp`
at its wavelengths, shifted by the shift, and raised to the scale times the ratio of
their airmasses (Beer's law: AIRMASS in the headers, or `--airmass`). Values of the
calibration below `--threshold` are taken to be it. The output is multiplied by the
mean of the correction, so that its level stays where it was. Only the first band is
corrected; the others are copied.

The shift starts at `--shift` and the scale at `--scale`. With `--xcorr` the two are
cross-correlated over `--lag` pixels, their continua taken out and their ends tapered,
for the shift; with `--tweakrms` the shift and then the scale are moved, by steps from
`--dshift` and `--dscale` halved until they are below 0.01, to where the RMS of the
corrected spectrum over `--sample` is least -- the RMS of each pixel less the mean of
those three either side, which is what a residual band leaves. What is found for one
spectrum is where the next one starts. The log gives the shift, the scale and the
normalization, and warns of pixels beyond the calibration's last, which take its last
value.

By eye (`--interactive`) each spectrum opens in IRAF's graphs after the search: above,
three candidates -- the spectrum corrected with the current shift and scale in the
middle, and with a step less and a step more of one of them below and above, set apart
by the offset; below, the calibration where the spectrum is. The keys are IRAF's, and
act where the mouse is:

| key | what it does |
|-----|--------------|
| `x` | the candidates by shift; over one of them, that one is taken |
| `y` | the candidates by scale; over one of them, that one is taken |
| `a` | the shift and scale of least RMS over the sample, from where they are |
| `e` | double the step of the candidates shown |
| `s` | a sample region from here to the next key: `s` adds it, `n` starts anew |
| `c` `d` | show or hide the calibration, or the spectrum, below |
| `r` | redraw |
| `w` | gtools' window keys, on the graph the mouse is over; the two share their x axis |
| `q` | done: the middle candidate's shift and scale are the spectrum's |
| `:shift` `:scale` `:dshift` `:dscale` `:offset` `:smooth` `:sample` | show a value, or set it |

Taking the middle candidate with the key of the graph shown halves its step, so the
choice closes in: `y` on the better of three scales until the middle is the best, the
same with `x` for the shift. The title says where things stand -- scale, shift, the step
of the one shown, the offset and the RMS. Before each spectrum after the first the
window asks whether to go on by eye; "No to all" leaves the rest to the search. Beside
IRAF's graph, each candidate says its shift or scale at its right end and the sample is
shaded.

Without a window, `--cursor` reads the keys from a file in IRAF's cursor format
(`x y wcs key` or `x y wcs :command`; wcs 1 the candidates, 2 the graph below), on from
one spectrum to the next, and `--answers` the replies to "Search interactively?" (no,
yes, NO, YES), a line for each spectrum asked:

```console
$ cat keys.cur
6900 2.0 1 y
6900 1.0 1 y
0 0 1 :shift
0 0 1 q
0 0 1 q
$ harwood telluric s c t --cursor keys.cur
ok telluric in 0.70s
s[*,1]: shift 0.7001734
TELLURIC:
  Output: c - spectrum s
  Input: s[*,1] - star one
  Calibration: t[*,1] - tell one
  Tweak: shift = 0.70, scale = 1.185, normalization = 0.9715
  WARNING: 1 pixels outside of calibration limits
s[*,2]:   Input: s[*,2] - star two
  Calibration: t[*,2] - tell two
  Tweak: shift = 0.70, scale = 1.013, normalization = 0.9735
  WARNING: 1 pixels outside of calibration limits
  -> c.fits
```

IRAF 2.18's telluric and skytweak are held to the bit: twenty-seven sessions -- the
shift and scale found by cross-correlation and the least RMS, each alone or given; the
lag, the sample, the steps; the calibration's apertures by number, ignored or missing;
thresholds, and a calibration that goes to zero; the airmass from the header or given;
a calibration that covers part of the spectrum; in place; four bands; two inputs with
one calibration or one each -- say the same, line for line, and leave the same pixels.
By eye, twenty-one more from cursor files: every key and colon command, the prompts and
the bells, the answers, and each graph drawn -- its windows, title, curves and sample
marks where IRAF's metacode has them.

What IRAF does by eye that is worth knowing, kept: `w` over the candidates measures
its steps (`u`, `d`, `y`, `z`...) on the lower graph's window when there is one, the
last drawn; once the lower graph has been shown the candidates' x axis is never
numbered again, even alone; with no step in the shift, `x` cannot be chosen (the code
asks about the shift's step twice, never the scale's), and with no step in the scale
`y` shows three alike; on an x axis run backwards the sample's ticks are not drawn.
One thing is not kept: `:sample` with a text that is no range. IRAF means to refuse it
and takes it -- no pixel is then in the sample, the RMS is INDEF, and the search finds
nothing for that spectrum or the next. harwood warns and keeps the sample it had.
