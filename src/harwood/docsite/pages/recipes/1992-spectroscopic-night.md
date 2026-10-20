# A night of spectra, 25 January 1992

Twenty-five frames from the KPNO coudé feed, taken on one night and used as the
teaching set for the course's exercise 5. Two early-type binaries, DH Cep and
AH Cep, at a grating setting labelled 6707, with a comparison lamp at each end
of the night.

This recipe takes them from raw frames to extracted, traced, time-stamped
spectra with their arcs assigned -- and then stops, at the one step where the
software cannot decide for you. That stop is not a gap in the recipe. It is the
most important thing in it, and §"Where it stops" says why.

```
data      ~/iraf/exercises/spec/fsp0001 .. fsp0025
detector  Tektronix 1024 (te1k), 133 x 1024 raw
site      Kitt Peak, coudé feed
gain      5.4 e-/ADU, read noise 5 e-   (from the header, not assumed)
```

---

## 0. A place to work

```console
$ mkdir night && cd night && mkdir raw
$ cp ~/iraf/exercises/spec/fsp00* raw/          # rename to .fits as you copy
$ harwood init .
Created harwood.yaml
```

`harwood.yaml` holds the parameters for this reduction. Nothing is written to
your home directory, and nothing carries over from another reduction -- which
is the difference from IRAF's `uparm`, where a parameter set three months ago
in another directory is still in force and invisible.

## 1. Look at what you have

```console
$ harwood ccdlist 'raw/sp00*.fits'
```

| image | type | shape | exptime | object |
|---|---|---|---|---|
| raw/sp0001.fits | zero | 133x1024 | 0 | biases 1st afternoon |
| … | | | | |
| raw/sp0014.fits | flat | 133x1024 | 60 | flats-6707 |
| raw/sp0018.fits | comp | 133x1024 | 60 | comp 6707 start of night |
| raw/sp0019.fits | object | 133x1024 | 1200 | DHCep 6707 |
| raw/sp0025.fits | comp | 133x1024 | 60 | comp 6707 |

Ten biases, twilight flats, four dome flats, an arc, three exposures of DH Cep,
three of AH Cep, and a second arc. The two arcs are three hours apart, and that
is deliberate: a dispersion solution belongs to a moment, not to a
spectrograph.

The frames are 133 columns wide and only 101 of those are illuminated. Columns
102 to 133 are the overscan -- the amplifier reading out with the shutter shut,
which measures the electronic zero at the moment of the exposure rather than
hours earlier.

## 2. Bias

```console
$ harwood ccdproc 'raw/sp000[1-9].fits' 'raw/sp0010.fits' --output b \
      --overscan --trim --biassec '[102:133,1:1024]' --trimsec '[1:101,1:1024]'
$ harwood zerocombine 'b/sp00*.fits' Zero.fits
ok zerocombine in 0.73s (numba 0.67.0)
  images: 10
  rejected: 103424
  mean_level: 0.1633438640629941
  -> Zero.fits
```

One value rejected per pixel, 101 × 1024 of them: the highest of the ten,
as IRAF's zerocombine does. The mean level of 0.16 counts is the point: after the overscan has taken out
the electronic zero, what is left in a bias frame is structure, not offset. If
this came out at 500 the overscan had not been subtracted, and everything
downstream would be wrong by a constant nobody would notice.

Add `--explain` to any of these to see the arithmetic:

```console
$ harwood zerocombine 'b/sp00*.fits' Zero.fits --explain
```

## 3. Everything else

```console
$ harwood ccdproc 'raw/sp001[4-9].fits' 'raw/sp002[0-5].fits' --output p \
      --overscan --trim --zerocor --zero Zero.fits \
      --biassec '[102:133,1:1024]' --trimsec '[1:101,1:1024]'
```

Twelve frames: the flats, both arcs, and all six object exposures. Each comes
out 101 x 1024, with `OVERSCAN`, `TRIM` and `ZEROCOR` marks in its header so
running this again corrects nothing twice.

## 4. When each exposure happened

```console
$ harwood setairmass 'p/sp00*.fits' --observatory kpno
```

| image | utmiddle | airmass | begin | end | was |
|---|---|---|---|---|---|
| sp0018.fits | 01:22:50.0 | 1.1269 | 1.1266 | 1.1272 | 1.12 |
| sp0019.fits | 02:18:22.0 | 1.5654 | 1.5229 | 1.6100 | 1.52 |
| sp0021.fits | 03:00:01.0 | 1.7792 | 1.7207 | 1.8408 | 1.72 |
| sp0024.fits | 04:02:59.0 | 2.1084 | 2.0405 | 2.1793 | 2.04 |

```
● note  6 of 12 frames had an airmass in the header that differs from the
   effective one by more than 0.01 -- sp0024.fits had 2.04 and travelled
   through 2.108.
   -> try: nothing -- the new value is the better one.
```

Look at the `begin` column against `was`. They agree to 0.004 across a factor
of two in airmass, from nothing but the site, the coordinates and the string
`25/01/92`. That agreement is worth pausing on: it means the whole time chain
-- date parsing, sidereal time, refraction -- is right, checked against a
number somebody else's software wrote thirty years ago.

And then look at `begin` against `airmass`. The 1992 system recorded the
airmass at the moment the shutter opened. Over 1200 seconds DH Cep sank from
1.52 to 1.61, and the light that actually landed on the detector came through
a mean of 1.565. That difference is why `setairmass` exists, and using 1.52 in
a flux calibration takes out too little extinction.

```console
$ harwood setjd 'p/sp00*.fits'
```

Julian dates, plus the heliocentric correction -- up to 16.6 minutes of light
travel time across the Earth's orbit, which for a 2.1-day binary is not a
rounding error.

## 5. Extract

```console
$ harwood apall p/sp0019.fits sp0019.ms.fits \
      --gain 5.4 --readnoise 5 --nsum 20 --t-step 20 --weights variance
ok apall in 1.35s
  apertures: 1
  centres: 53.95
  limits: -2.90:+2.87
```

Repeat for `sp0020` through `sp0024`. The centres come out 53.95, 53.93, 53.94,
53.81, 53.85, 53.85 -- six exposures over two hours agreeing to 0.14 pixels,
which says the guiding held and the extraction found the same star each time.

Three things worth knowing about what that command did:

- **`--weights variance`** is Horne's optimal extraction: the spatial profile
  is modelled along the dispersion, and each row across the slit is then
  weighted by that profile and by its variance rather than simply summed.
  These are 1200-second exposures at a signal-to-noise of 88 per pixel: the
  noise is photons, and there is little for the weighting to recover -- on a
  synthetic spectrum that bright, the plain sum scatters under one per cent
  more. It earns its keep on a faint spectrum, where the read noise at the
  edges of the profile is what a plain sum is throwing away: fifty times
  fainter, the plain sum scatters 48 per cent more, measured as scatter
  about a known answer rather than as a claimed error bar. It is also the
  only extraction with an uncertainty for every point.
- **The trace.** The spectrum does not sit on one column; the camera has
  distortion and the instrument flexes. Here it drifts −1.51 pixels end to end,
  fitted to 0.017 pixels rms (`--explain` prints both). The trace goes to
  `database/apsp0019`.
- **`--gain 5.4`** comes from the header. Guessing it -- 3 is a common default
  -- would leave every uncertainty in the file overstated by 35 per cent, and
  nothing would look wrong.

To set the apertures by hand first:

```console
$ harwood apall p/sp0019.fits sp0019.ms.fits --edit
```

That opens the profile with the apertures drawn on it, IRAF's keys and all.
To see the parameters in an editor instead of remembering them:

```console
$ harwood apall --editpar
```

## 6. Which arc calibrates which exposure

```console
$ harwood refspectra 'sp00*.ms.fits' \
      --references p/sp0018.fits --references p/sp0025.fits
```

| spectrum | arc | weights |
|---|---|---|
| sp0019.ms.fits | sp0018 + sp0025 | 0.684 0.316 |
| sp0020.ms.fits | sp0018 + sp0025 | 0.565 0.435 |
| sp0021.ms.fits | sp0018 + sp0025 | 0.446 0.554 |
| sp0022.ms.fits | sp0018 + sp0025 | 0.325 0.675 |
| sp0023.ms.fits | sp0018 + sp0025 | 0.206 0.794 |
| sp0024.ms.fits | sp0018 + sp0025 | 0.088 0.912 |

Each exposure gets **both** arcs, weighted by how close in time each one is.
That is the default, `--select interp`, and it is why the night has two arcs
rather than one: between them the instrument's flexure is smooth, so an
exposure in the middle is better served by an interpolation than by whichever
arc happens to be nearer.

The assignment is written into each spectrum's own header, as `REFSPEC1` and
`REFSPEC2` in IRAF's form. From here on the file says which arc calibrates it,
and still says so next year.

How much flexure was there, in fact? The two arcs' lines sit 0.23 pixels apart
over three hours. That is very little -- a coudé feed lives in a fixed room and
does not tilt with the telescope -- so on *this* night `--select nearest` would
have given nearly the same answer. On a Cassegrain spectrograph it would not.

## 7. Where it stops

The arcs, like the stars, run along the columns of these frames, so
`identify` is told to read the middle column rather than its default, the
middle line -- which here would be 101 pixels across the slit, and harwood
would say so:

```console
$ harwood identify p/sp0018.fits --coordlist 'linelists$thar.dat' --section 'middle column'
✖ identify stopped  p/sp0018.fits[51,*]: no approximate wavelength or dispersion to start from
   -> try: give --crval (the wavelength at the middle) and --cdelt (per pixel), good to about one
           per cent -- from the observing log or a previous night; or mark four or five lines with
           --interactive; or `autoidentify p/sp0018.fits INDEF INDEF` for the blind search, which
           on a dense list takes tens of minutes
```

**This is the correct answer, and it is the most important step in the recipe.**

The arc has 82 lines standing above its noise. A thorium-argon list has 277
lines between 6400 and 7100 Å -- one every two and a half Ångströms. Against
a list that dense, several quite different dispersions each explain a couple
of dozen lines, and a cubic will fit any of them to a tenth of a pixel. The
fit looks excellent and the answer is invented.

The blind search exists -- `harwood autoidentify p/sp0018.fits INDEF INDEF`,
IRAF's autoidentify with nothing to start from -- and on this arc it ran for
half an hour without an answer before it was stopped. There is a reason IRAF's
`identify` asks a person.

Two ways forward, and both need a person:

```console
# You know roughly where the spectrum sits -- from the grating setting,
# the observing log, or a previous night:
$ harwood identify p/sp0018.fits --coordlist 'linelists$thar.dat' \
      --section 'middle column' --crval 6707 --cdelt 0.15

# You do not, so you mark four or five lines yourself:
$ harwood identify p/sp0018.fits --coordlist 'linelists$thar.dat' \
      --section 'middle column' --interactive
```

The first refuses too, if the guess is wrong -- and 6707, the grating
setting written in the arc's title, is not where this spectrum is centred:

```
✖ identify stopped  p/sp0018.fits[51,*]: No solution found.
   -> try: check --crval and --cdelt against the lamp's range -- the OBJECT name of an arc is the
           grating setting, not necessarily the wavelengths it covers -- and that --coordlist is
           that lamp's list; or mark a few lines by hand with identify --interactive.
```

A tenth of a pixel is how well an arc line centre can be measured. A solution
that scatters ten times that has not found a worse answer to the same question
-- it has answered a different one. So nothing is written.

This is what people mean when they say IRAF's `identify` is interactive
"because it is old". It is interactive because the problem requires it. What
harwood adds is not an automatic answer; it is refusing to fabricate one, and
saying why.

## 8. What comes after, once you have the solution

```console
$ harwood dispcor sp0019.ms.fits sp0019.w.fits
```

No `--reference` is needed: step 6 wrote the answer into the header, and
`dispcor` combines the two arcs' solutions in the proportion `refspectra`
recorded.

It then checks itself, by looking up its own strongest features in the line
catalogue and reporting which ones landed on a real wavelength. That check is
worth more than the arc's rms: the rms says the fit reproduces the lines it was
*given*, which a solution in entirely the wrong place does beautifully. Real
features landing on real wavelengths is a separate claim, and it is the one
that catches a solution five Ångströms out.

Then, to see what the spectrum is made of:

```console
$ harwood lineid sp0019.w.fits --features absorption --plot
```

That names each feature -- the species, the ion, the band -- says which are
ambiguous, and works out whether the lines agree on a radial velocity. The
`--plot` opens the spectrum with every catalogued line drawn where it falls,
coloured by what produces it, so an absorption that is oxygen above the
telescope does not look like one in the star.

*(Neither command has been run on this night: both need the dispersion
solution that §7 stops at. The output above describes what they do, not what
they printed.)*

## 9. Where this night ends

The flux chain -- `standard`, `sensfunc`, `calibrate` -- cannot be run on these
frames, because **no flux standard was observed that night**. `harwood ccdlist`
shows biases, flats, two arcs and six programme exposures, and nothing else. It
is not a limitation of the software and no parameter fixes it; a calibration
that was not taken cannot be applied.

What you can do with what is here: measure line positions and equivalent widths
against a wavelength scale, and compare the six exposures with each other. What
you cannot do: quote a flux in erg/cm²/s/Å. For that the night needed twenty
minutes on a Feige or a BD star.

## 10. What was done, recoverable

```console
$ harwood log
```

| # | task | outputs |
|---|---|---|
| 1 | ccdlist | - |
| 2 | ccdproc | b/sp0001.fits and 9 more |
| 3 | zerocombine | Zero.fits |
| 4 | ccdproc | p/sp0014.fits and 11 more |
| … | | |
| 7 | apall | sp0019.ms.fits, database/apsp0019 |
| … | | |
| 13 | refspectra | sp0019.ms.fits and 5 more |

```console
$ harwood log apall --full    # every parameter of every apall run, and where it came from
$ harwood log --replay 2      # the command that repeats run 2, exactly
harwood ccdproc 'raw/sp000[1-9].fits' raw/sp0010.fits --output b --overscan --trim --biassec '[102:133,1:1024]' --trimsec '[1:101,1:1024]'
```

A run that was not told where to write also leaves a directory under
`output/`, named `hrwd_<task>_<date>`, with a `run.yaml` recording the
parameters, their origins and the outputs written. Between that, the log, the `HISTORY`
cards in each file, and `--explain` on any command, there should be no step in
this recipe you cannot reconstruct six months from now -- including the ones
that failed.
