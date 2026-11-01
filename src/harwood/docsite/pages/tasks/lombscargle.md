Finds the periods in a light curve, the Lomb-Scargle periodogram (Zechmeister & Kuerster 2009,
VanderPlas 2018): at each frequency a sinusoid is fitted at the times there are, each point weighted
by its error, and the power is how much of the scatter it explains. It wants no even sampling. IRAF
has `pdm`, phase dispersion minimization, from before it. The arithmetic is astropy.timeseries, which
everyone uses: what is here is what a peak is worth.

```console
$ harwood lombscargle star.txt --time 1 --value 2 --error 3 --folded star.fold --graph star.svg
```

The table is any harwood reads, photometry's too, and `--time`, `--value` and `--error` say which
columns, by name or by number from 1. What has no value, IRAF's `INDEF` too, is left out, and it says
how many. The times can be in any unit, and the periods are in that unit.

**What it finds.** The highest peaks, no two within a peak's width, with their period, power,
amplitude and false alarm probability: Baluev's, how often noise would make one so high somewhere in
the range looked at. `--minperiod` and `--maxperiod` say the range, the span of the times being the
longest by default and twice the median gap the shortest. Measures taken at random reach well beyond
that, and `--minperiod` goes there. `--harmonics` fits more terms at each frequency, for a curve that
is no sinusoid; an RR Lyrae's fundamental is the highest peak then, where with one term it can be a
harmonic. With more than one term there is no such bound, and the probability is left empty.

**What it warns of.**

- A peak of the sampling. A constant at the same times has its own periodogram, and a nightly gap puts
  a peak in it at one cycle per day. A peak that far from the best in frequency is named in the
  `alias` column, and if it is as high it says the data cannot tell which is the star's.
- A false alarm probability above `--fap`: noise makes one that high.
- A best period above half the span: fewer than two cycles cannot show that it repeats.

`--output` writes the whole periodogram, `--graph` draws it, and `--folded` writes phase and value at
the best period, from the first time.

Not IRAF's, so there is no IRAF to compare with. It is held to the sum in the definition, done by
hand at each frequency, and to periods put there beforehand, in a gapped sampling, with their
amplitude. It has not been run on a real light curve: the tutorial's data has none.
