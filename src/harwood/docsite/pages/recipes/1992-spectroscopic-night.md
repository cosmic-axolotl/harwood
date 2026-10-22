# A night of spectra, 25 January 1992

Twenty-five frames from the KPNO coudé feed, taken on one night and used as the
teaching set for the course's exercise 5. Two early-type binaries, DH Cep and
AH Cep, at a grating setting labelled 6707, with a comparison lamp at each end
of the night.

This recipe takes them from raw frames to wavelength-calibrated spectra with
their lines named, the way the course's own exercise does it. On the way it
comes to the one step where the software cannot decide for you -- which part of
the spectrum the arc covers -- and §8 is about why it refuses to guess, and
where the answer actually comes from.

```
data      ~/iraf/exercises/spec/fsp0001 .. fsp0025
detector  Tektronix 1024 (te1k), 133 x 1024 raw
site      Kitt Peak, coudé feed
gain      5.4 e-/ADU, read noise 3.5 e-   (GAIN and RDNOISE, in every header)
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

Each also comes out with its uncertainty beside it. A harwood image holds
three planes -- `SCI`, the data; `VAR`, its variance; `DQ`, why a pixel is not
to be trusted -- and the variance is born here: with the overscan and the bias
off, what is left counts the electrons each pixel collected, so its variance is
data/gain + (read noise/gain)², worked out from `GAIN` and `RDNOISE` in the
header. Everything downstream carries it. To take the frames to IRAF instead,
`--format iraf` writes the data alone in one HDU, as IRAF reads it, and says
that the variance stayed behind.

## 4. When each exposure happened

```console
$ harwood setairmass 'p/sp00*.fits' --observatory kpno
```

| image | utmiddle | airmass | begin | end | was |
|---|---|---|---|---|---|
| sp0018.fits | 01:22:50.0 | 1.1264 | 1.1261 | 1.1267 | 1.12 |
| sp0019.fits | 02:18:22.0 | 1.5645 | 1.5221 | 1.6091 | 1.52 |
| sp0021.fits | 03:00:01.0 | 1.7780 | 1.7196 | 1.8395 | 1.72 |
| sp0024.fits | 04:02:59.0 | 2.1075 | 2.0396 | 2.1783 | 2.04 |

```
● note  6 of 12 frames had an airmass in the header that differs from the
   effective one by more than 0.01 -- sp0024.fits had 2.04 and travelled
   through 2.107.
   -> try: nothing -- the new value is the better one.
```

Look at the `begin` column against `was`. They agree to better than 0.01 --
the header only carries two decimals -- across a factor of two in airmass, from nothing but the site, the coordinates and the string
`25/01/92`. That agreement is worth pausing on: it means the whole time chain
-- date parsing, sidereal time, refraction -- is right, checked against a
number somebody else's software wrote thirty years ago.

And then look at `begin` against `airmass`. The 1992 system recorded the
airmass at the moment the shutter opened. Over 1200 seconds DH Cep sank from
1.52 to 1.61, and the light that actually landed on the detector came through
a mean of 1.564. That difference is why `setairmass` exists, and using 1.52 in
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
      --gain GAIN --readnoise RDNOISE --nsum 20 --t-step 20 --weights variance
ok apall in 0.58s
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
  These are 1200-second exposures at a signal-to-noise of 89 per pixel: the
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
- **`--gain GAIN --readnoise RDNOISE`** names the header keywords rather
  than copying numbers out of them: 5.4 e-/ADU and 3.5 e-. Guessing the gain
  -- 3 is a common default -- would leave every uncertainty in the file
  overstated by 35 per cent, and nothing would look wrong. (An earlier
  version of this recipe typed `--readnoise 5`, "from the header"; the header
  says 3.5. Naming the keyword is how that mistake stops being possible.)

To set the apertures by hand first:

```console
$ harwood apall p/sp0019.fits sp0019.ms.fits --edit
```

That opens the profile with the apertures drawn on it, IRAF's keys and all.
To see the parameters in an editor instead of remembering them:

```console
$ harwood apall --editpar
```

## 6. The arcs, extracted with a star's aperture

An arc lamp fills the slit evenly: there is no star in it to find and nothing
to trace. So it is extracted with the apertures of a frame that has one --
here DH Cep's first exposure -- the same centre, the same limits and the same
trace, so the lamp's lines are measured on exactly the pixels the star's light
was summed from.

```console
$ harwood apall p/sp0018.fits sp0018.ms.fits --references sp0019 \
      --no-recenter --no-resize --no-trace --background none
ok apall in 0.20s
  apertures: 1
  centres: 53.95
  limits: -2.90:+2.87
  -> sp0018.ms.fits
  -> database/apsp0018
$ harwood apall p/sp0025.fits sp0025.ms.fits --references sp0019 \
      --no-recenter --no-resize --no-trace --background none
```

`--references sp0019` takes `database/apsp0019`, which §5 wrote. Recentering,
resizing and tracing are all off because there is no star to recentre on, and
the background is off because the lamp *is* the background: subtracting it
would leave nothing. These are the course's own settings.

The exercise extracts both arcs with one star's aperture, and so does this. For
radial velocities you would extract each arc with each star's own aperture --
`--references sp0022` for AH Cep -- since a star sitting on another column
sees the lamp through a slightly different part of the optics. Here all six
centres agree to 0.14 pixels, and one aperture does for all.

## 7. Which arc calibrates which exposure

```console
$ harwood refspectra 'sp0019.ms.fits,sp002[0-4].ms.fits' \
      --references sp0018.ms.fits --references sp0025.ms.fits
```

| spectrum | arc | weights |
|---|---|---|
| sp0019.ms.fits | sp0018.ms + sp0025.ms | 0.684 0.316 |
| sp0020.ms.fits | sp0018.ms + sp0025.ms | 0.565 0.435 |
| sp0021.ms.fits | sp0018.ms + sp0025.ms | 0.446 0.554 |
| sp0022.ms.fits | sp0018.ms + sp0025.ms | 0.325 0.675 |
| sp0023.ms.fits | sp0018.ms + sp0025.ms | 0.206 0.794 |
| sp0024.ms.fits | sp0018.ms + sp0025.ms | 0.088 0.912 |

Each exposure gets **both** arcs, weighted by how close in time each one is.
That is the default, `--select interp`, and it is why the night has two arcs
rather than one: between them the instrument's flexure is smooth, so an
exposure in the middle is better served by an interpolation than by whichever
arc happens to be nearer.

The references are the *extracted* arcs, because that is what will carry the
dispersion solutions: `dispcor` looks for `database/idsp0018.ms`, the name of
the file `identify` read. The assignment is written into each spectrum's own
header, as `REFSPEC1` and `REFSPEC2` in IRAF's form. From here on the file says
which arc calibrates it, and still says so next year.

## 8. Which wavelengths the arc covers

`identify` needs to know roughly where the spectrum sits before it can name
a single line. Asked with nothing to go on, it stops:

```console
$ harwood identify sp0025.ms.fits --coordlist 'linelists$thorium.dat'
✖ identify stopped  sp0025.ms.fits - Ap 1: no approximate wavelength or dispersion to start from
   -> try: give --crval (the wavelength at the middle) and --cdelt (per pixel), good to about one
           per cent -- from the observing log or a previous night; or mark four or five lines with
           --interactive; or `autoidentify sp0025.ms.fits INDEF INDEF` for the blind search, which
           on a dense list takes tens of minutes
```

**This is the correct answer, and it is the most important step in the recipe.**

The arc shows over a hundred lines. A thorium-argon list holds a line every
couple of Ångströms. Against a list that dense, several quite
different dispersions each explain a couple of dozen lines, and a cubic will fit
any of them to a tenth of a pixel. The fit looks excellent and the answer is
invented. The blind search exists -- `harwood autoidentify sp0025.ms.fits INDEF
INDEF`, IRAF's autoidentify with nothing to start from -- and on this night's
arc it ran for half an hour without an answer before it was stopped. There is a reason
IRAF's `identify` asks a person.

The title of every frame offers a number, 6707, and it is the wrong one:

```console
$ harwood identify sp0025.ms.fits --coordlist 'linelists$thorium.dat' --crval 6707 --cdelt 0.15
✖ identify stopped  sp0025.ms.fits - Ap 1: No solution found.
   -> try: check --crval and --cdelt against the lamp's range -- the OBJECT name of an arc is the
           grating setting, not necessarily the wavelengths it covers -- and that --coordlist is
           that lamp's list; or mark a few lines by hand with identify --interactive.
```

A tenth of a pixel is how well an arc line centre can be measured. A solution
that scatters ten times that has not found a worse answer to the same question
-- it has answered a different one. So nothing is written.

The answer comes from where it came from in 1992: a person who knew. The course
hands out a chart of this very arc (`docs/spec.eps`, IRAF's own `identify` plot
of it, "comp 6707"), with the lines labelled from 4445.90 to 4676.06 Å. So the
spectrum is centred near 4560 Å and covers about 230 Å in 1024 pixels -- 0.225
Å per pixel -- and it runs backwards: the exercise tells you to press `f` to
flip it so the wavelengths increase to the right. That is a negative
dispersion.

```console
$ harwood identify sp0025.ms.fits --coordlist 'linelists$thorium.dat' \
      --crval 4560 --cdelt -0.225 --function chebyshev --order 3 \
      --niterate 5 --low-reject 2.5 --high-reject 2.5
ok identify in 3.01s
  features: 122
  used: 100
  rms: 0.02887
  dispersion: 0.23092
  range: 4444.40 to 4680.63 Angstroms
  -> database/idsp0025.ms
```

A Chebyshev of order 3, as the course asks. 122 thorium lines placed, 100 kept,
an rms of 0.029 Å -- an eighth of a pixel, which is about what a line centre is
worth. The 22 left out are blends in a list this dense: 20 of them are lines IRAF's
own autoidentify rejects on this arc, and it never placed the other two. An order 4 gives the same 0.029: the
polynomial is not what limits the fit.

The course asks for "RMS ≈ 0.01 Å", reached there by deleting lines by hand.
Here is what each choice of rejection does on this arc:

| rejection | lines kept | rms |
|---|---|---|
| none | 122 of 122 | 0.112 Å |
| 3 σ, 2 iterations (the course's `:nite 2`) | 115 | 0.058 Å |
| 2.5 σ, 5 iterations | 100 | 0.029 Å |
| 2 σ, 10 iterations | 70 | 0.009 Å |

The last row is the course's number, bought by throwing away 52 of the 122 lines.
Each pass recomputes σ from what survived and cuts again, so the rms falls
because there is less left to measure, not because the solution got better. The
interactive way (`--interactive`) is the honest route to a lower number: you
look at each line you delete.

## 9. The other arc

The arc from the start of the night does not need a person: the one just
identified tells `reidentify` where every line should be.

```console
$ harwood reidentify sp0025.ms sp0018.ms --coordlist 'linelists$thorium.dat'
  Reference image = sp0025.ms, New image = sp0018.ms, Refit = yes
          Image Data    Found     Fit Pix Shift  User Shift  Z Shift      RMS
sp0018.ms - Ap 1      116/122  95/116     0.224     -0.0518  -1.1E-5    0.024
```

116 of the 122 lines found again, the fit refitted from them to 0.024 Å, and
the answer to a question worth asking: how far did the spectrograph move in
three hours? **0.224 pixels**, or 0.05 Å. That is very little -- a coudé feed
lives in a fixed room and does not tilt with the telescope -- so on *this*
night calibrating every exposure with the nearer arc would have given nearly
the same answer. On a Cassegrain spectrograph it would not.

## 10. Wavelengths on the spectra

First the grid, without writing anything, as the course does:

```console
$ harwood dispcor 'sp0019.ms.fits,sp002[0-4].ms.fits' --global --listonly
sp0019.ms: REFSPEC1 = 'sp0018.ms 0.68366086'
sp0019.ms: REFSPEC2 = 'sp0025.ms 0.31633914'
sp0019.ms.w: ap = 1, w1 = 4444.408, w2 = 4680.665, dw = 0.230945, nw = 1024
…
sp0024.ms.w: ap = 1, w1 = 4444.408, w2 = 4680.665, dw = 0.230945, nw = 1024
```

`--global` gives all six the same grid -- the widest range any of them reaches
at the finest dispersion -- so they can be compared pixel for pixel. Then for
real:

```console
$ harwood dispcor 'sp0019.ms.fits,sp002[0-4].ms.fits' \
      'sp0019.w.fits,sp0020.w.fits,sp0021.w.fits,sp0022.w.fits,sp0023.w.fits,sp0024.w.fits' \
      --global --explain
  …
  6. · check: the strongest features, by name
       strongest features 5   on a catalogued line 3   landing on Si III 4567.84, He II 4541.59, He I 4471
  ▲ warning  18 of 1024 pixels fall outside the range the arc lines covered (12 to 1018); their
     wavelengths are extrapolated, and a polynomial extrapolates badly.
```

Each spectrum's solution is the two arcs' combined in the weights of §7.

**It checks itself.** The arc's rms says the fit reproduces the lines it was
*given*, which a solution in entirely the wrong place does beautifully. So
`dispcor` looks up the strongest features of the result in the line catalogue:
three of DH Cep's five land on He I 4471, He II 4541 and Si III 4568. He II
only appears in the hottest stars, and DH Cep is a pair of O stars -- real
features of the right star on real wavelengths, an independent check.

**The ends.** The arc's lines reach from pixel 12 to 1018; the 18 pixels
beyond are wavelengths extrapolated from a polynomial, and the warning says
so rather than letting them look as good as the rest.

## 11. What the spectra show

```console
$ harwood lineid sp0019.w.fits --features absorption
```

| wavelength | species | line | offset |
|---|---|---|---|
| 4471.43 | He I | He I 4471 | -0.053 |
| 4541.17 | He II | He II 4541.59 | -0.422 |
| 4541.85 | He II | He II 4541.59 | 0.262 |

DH Cep shows helium, neutral and ionised: an O star. He II comes out twice,
0.7 Å apart, and that is not two stars: these lines are four or five
Ångströms wide, and at a signal-to-noise of 89 the bottom of a line that broad
can show two minima from the noise alone. The proof is how fragile it is: with
the read noise mistyped as 5 e- the extraction weights moved a little, and He I
split in two while He II's minima moved by a quarter of an Ångström.

AH Cep's first exposure is a different matter:

```console
$ harwood lineid sp0022.w.fits --features absorption
● note  4 of 4 features are not in the catalogue, which holds only the lines that name themselves.
   Unidentified here means unnamed, not unreal.
```

| wavelength | line |
|---|---|
| 4467.97 | unidentified |
| 4475.82 | unidentified |
| 4644.88 | unidentified |
| 4646.24 | unidentified |

Nothing named -- and the first two sit 3.5 Å either side of He I 4471.48,
7.85 Å apart. Forty minutes later, in `sp0023`, they are at 4468.19 and
4475.45: 7.26 Å apart, and closing. That is AH Cep being what it is, a pair of
B stars in a 1.77-day orbit, seen as two spectra moving in opposite directions.
`lineid` leaves them unnamed because each is about four Ångströms from the
laboratory wavelength, beyond its tolerance. Tell it the velocity, and each
becomes He I in turn:

```console
$ harwood lineid sp0022.w.fits --features absorption --velocity -235
│ 4467.97    │ absorption │ He I    │ He I 4471     │ -0.009 │ yes  │
$ harwood lineid sp0022.w.fits --features absorption --velocity 291
│ 4475.82    │ absorption │ He I    │ He I 4471     │ 0      │ yes  │
```

One star approaching at 235 km/s, the other receding at 291 -- measured
against the laboratory, so their mean, +28 km/s, is the system's own motion
plus whatever the calibration is off by. And look at what
happens to the pair near 4645 Å at the same time: at −235 one of them becomes
C III 4650, at +291 both become N III 4641. A catalogue offers *some* line near
almost any wavelength; that is why a name on its own is not evidence, and why
He I counts here -- it is the strongest line a B star has, and both components
land on it.

## 12. Where this night ends

The flux chain -- `standard`, `sensfunc`, `calibrate` -- cannot be run on these
frames, because **no flux standard was observed that night**. `harwood ccdlist`
shows biases, flats, two arcs and six programme exposures, and nothing else. It
is not a limitation of the software and no parameter fixes it; a calibration
that was not taken cannot be applied.

What you can do with what is here: measure line positions, radial velocities
and equivalent widths against a wavelength scale, and compare the six
exposures with each other. What you cannot do: quote a flux in erg/cm²/s/Å.
For that the night needed twenty minutes on a Feige or a BD star. The course
ends by combining each star's three exposures with `scombine`, which harwood
does not have yet -- and for two binaries whose lines move between exposures,
combining them would smear exactly what §11 found.

## 13. What was done, recoverable

```console
$ harwood log
```

| # | task | outputs |
|---|---|---|
| 2 | ccdproc | b/sp0001.fits and 9 more |
| 3 | zerocombine | Zero.fits |
| 4 | ccdproc | p/sp0014.fits and 11 more |
| … | | |
| 7 | apall | sp0019.ms.fits, database/apsp0019 |
| … | | |
| 13 | apall | sp0018.ms.fits, database/apsp0018 |
| 15 | refspectra | sp0019.ms.fits and 5 more |
| 16 | identify | database/idsp0025.ms |
| 17 | reidentify | database/idsp0018.ms |
| 19 | dispcor | sp0019.w.fits and 5 more |
| 20 | lineid | output/hrwd_lineid_…/sp0019.w.lines.csv |

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
