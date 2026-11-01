Finds a transit or an eclipse in a light curve, by box least squares (Kovacs, Zucker & Mazeh 2002): a
box is fitted at each period, duration and phase. A dip that is flat and short spreads its power over
many harmonics in [lombscargle](task:lombscargle), and a box keeps it in one place. IRAF has nothing
for this. The arithmetic is astropy.timeseries, which everyone uses.

```console
$ harwood bls star.txt 0.05,0.1,0.2 --time 1 --value 2 --error 3 --graph star.svg
```

The second argument is the lengths of the dip to try, in the times' unit, with commas. The columns
are as in [lombscargle](task:lombscargle). `--magnitudes` says the measures are magnitudes, where a dip
is a rise, and they are taken to fluxes first, the median being 1, so that the depth is a fraction.

It gives the best period, depth with its error, duration, the epoch of a dip's middle, and how many
dips have a measure in them. `--minperiod` is twice the longest duration by default, and `--maxperiod`
the longest period with `--transits` dips in the span.

**What it warns of.**

- Odd and even dips that differ by more than 3 sigma. An eclipsing binary of nearly equal stars looks
  like a transit at half its period, and this tells them apart: it says what period to try.
- Fewer dips with a measure in them than `--transits`.

`--output`, `--graph` and `--folded` are as in [lombscargle](task:lombscargle).

Not IRAF's, so there is no IRAF to compare with. It is held to dips put there beforehand, in a gapped
sampling: period, duration, depth within its error, epoch, and a binary with unequal eclipses found
at half its period and told so. It has not been run on a real light curve: the tutorial's data has
none.
