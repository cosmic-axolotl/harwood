Fits gaussian, lorentzian and voigt profiles to the lines of spectra -- one line or a
blend -- and gives each one's centre, the continuum under it, its flux, equivalent
width, peak and widths, with errors if the noise is known.

```console
$ harwood fitprofs a --region "4165 4210" --positions p2 --gfwhm 5 --lines 1 --nerrsample 60 --sigma0 6 --invgain 0.05 --output m --logfile ""
ok fitprofs in 1.59s
# Nov  6  7:40 a - Ap 1: a one
# Nfit=2, background=yes, positions=all, gfwhm=all, lfwhm=all
#   center      cont      flux       eqw      core     gfwhm     lfwhm
  4179.333  985.3342   2004.75    -2.035   325.885     5.779        0.
 (0.07764) (3.30475) (54.6761) (0.05796) (9.46634) ( 0.1952) (     0.)
   4192.95  992.3033   1190.69      -1.2   172.652     6.479        0.
 (0.14453) (3.30977) (73.5175) (0.07518) (7.87053) ( 0.4049) (     0.)
a[*,1](1)  -->  m(1)
  -> m.fits
$ harwood fitprofs @night --region "6540 6590" --positions halpha --profile voigt
```

`--positions` is a file of the profiles, one a line: the position, and if given the
peak (INDEF, or left out, for the data's there), the type (`gaussian`, `lorentzian`,
`voigt`) and the widths; what a line leaves out is `--profile`, `--gfwhm` and
`--lfwhm`. As IRAF reads it, a lorentzian's width is its fourth field only when the
line has all five -- otherwise `--gfwhm`, `--lfwhm` notwithstanding.

The fit is over `--region`, two wavelengths, above a straight background through the
data at its ends; `--background` gives either end another way: a value, `avg(w1,w2)`
or `med(w1,w2)` of the data there, scaled by a third number if given. The profiles are
fitted by least squares, each pixel the mean of three subsamples across it: the
background held and then fitted (unless `--no-fitbackground`), the positions fixed,
then moved together, then each its own (`--fitpositions`), the widths scaled together
and then each its own (`--fitgfwhm`, `--fitlfwhm`). `--components` chooses which
profiles go into the log and the output; `--lines` and `--bands` which apertures and
bands, a leading `!` for all but those.

With `--sigma0` and `--invgain` the noise is sqrt(sigma0^2 + invgain*data), and
`--nerrsample` fits of the model with that noise added give errors -- the 68th
percentile of their differences from the fit, in parentheses under each profile, with
15 samples or more. The log goes to the terminal (`--verbose`) and to `--logfile`;
`--output` gets the model, or with `--option difference` the data less the model.

IRAF 2.18.1's fitprofs is held to the bit: forty sessions -- gaussian, lorentzian and
voigt profiles alone, blended and mixed, every constraint on positions and widths, the
background fitted, held, given, averaged, a median or scaled, Monte-Carlo errors, the
components, outputs of the model and of the difference, existing outputs and
`--clobber`, apertures and bands, two inputs, the log file, and what IRAF warns of or
refuses -- say the same, line for line, and write the same pixels. A profile whose
starting peak is 0 -- one outside the region -- is 0/0, as in IRAF, and stays NaN,
printed as IRAF prints a NaN: 0.

Where IRAF 2.18.1 fails, harwood does not follow it. An output written with
`--verbose` from apertures of different dispersions corrupts IRAF's memory -- the
verify line holds a world coordinate structure that a later aperture frees -- and the
image is never written; harwood writes it. `--merge` into an existing output ends in a
segmentation violation, and harwood refuses it. `--plotfile`, IRAF's metacode, is not
harwood's.
