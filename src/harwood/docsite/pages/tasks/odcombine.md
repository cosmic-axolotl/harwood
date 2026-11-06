Combines spectra with imcombine's code: resamples each onto one dispersion, keeps
it only where it reaches, and scales, rejects and averages, medians or sums what is
left, masks and all.

```console
$ harwood odcombine a,b,c,d,e o --reject avsigclip --scale median --smasktype goodvalue --apertures 1

Nov  5 22:44: ODCOMBINE
  combine = average, scale = median, zero = none, weight = none
  reject = avsigclip, mclip = yes, nkeep = 1
  lsigma = 3., hsigma = 3.
  blank = 0.
  masktype = goodval, maskval = 0
                Images   Median  Scale   Offsets Maskfile
             a[*,1](1)  821.45  1.000         1 abpm[*,1](1)
             b[*,1](1)   1178.  0.697         2 bbpm[*,1](1)
             c[*,1](1)  968.71  0.848         0
             d[*,1](1)   632.7  1.298         2 dbpm[*,1](1)
             e[*,1](1)  907.92  0.905         1

  Output image = o.0001, ncombine = 5
  -> o.0001.fits
$ harwood odcombine @night o --group all --combine median --sigmas s
$ harwood odcombine star1,star2 o --w1 4000 --w2 5000 --dw 1 --log --nrejmasks nr
```

`--group apertures` (the default) combines each aperture number across the inputs
into an output of its own, `o.0001`, `o.0002`...; `--group all` every spectrum into
one; `--group images` the apertures of each input into an output of its own, as many
outputs as inputs. `--apertures` chooses which take part.

The output's dispersion is the first spectrum's with `--first`; otherwise it runs
over all the inputs' wavelengths in the smallest of their steps, unless `--w1`,
`--w2`, `--dw` and `--nw` say otherwise (any three of them fix the fourth), evenly in
log wavelength with `--log`. Each spectrum is resampled onto it, the mean of its
interpolant (`--interp`, poly5) over each pixel, and kept only from the first to the
last pixel it covers by more than 1% of one. The output runs from the first pixel
any spectrum reaches to the last -- shorter than the grid when the grid goes beyond
them -- and the log's Offsets say where each starts on it.

A bad pixel mask is named by BPM in an input's header: with `--smaskformat
bpmspectrum` (the default), a spectrum of mask values for each aperture on its own
dispersion; with `bpmpixel`, a line of mask values for each of the input's lines.
`--smasktype` says what in it is good -- `goodvalue` or `badvalue` `--smaskvalue`,
`goodbits` or `badbits` of it. The mask is resampled with its spectrum, and a pixel is
bad where that leaves more than a hundredth of a bad pixel in it.

`--scale`, `--zero` and `--weight` are `none`, the spectrum's `mode`, `median` or
`mean` (over `--statsec`: `[x1:x2]` of each spectrum, `input [..]`, `output [..]` of
the output's pixels, or `overlap`, where they all are), `exposure` (`--expname`),
`@file`, or `!KEYWORD`; relative to the first spectrum's. `--lthreshold` and
`--hthreshold` leave values out before anything else. `--reject` is imcombine's:
`minmax`, `sigclip`, `avsigclip`, `ccdclip`, `crreject`, `pclip`, with `--nkeep`,
`--mclip`, `--grow` (a radius in pixels) and the CCD noise model. Where nothing is
left the output is `--blank`. `--sigmas` writes the sigma about the combination as a
spectrum; `--bpmasks`, `--rejmasks`, `--nrejmasks` and `--expmasks` the masks
imcombine writes, as IRAF's pixel lists (`.pl`). The header gets IMCMBnnn, a card for
each spectrum combined, and NCOMBINE. `--outtype` is the output's pixel type; IRAF's
`double` combines in double precision, and harwood says so rather than combine in real
under that name.

IRAF 2.18.1's odcombine is held to the bit: seventy sessions -- every grouping, the
grid from the inputs and given, rising and falling, linear and logarithmic, the
first's, every rejection, nkeep and grow, every scaling, zero and weight from
statistics in each kind of section, a file or the header, thresholds, spectra that
reach part of the output or none of it, every kind and format of mask, the output's
integer types, and the sigma and the four masks -- write the same log line for line,
the same pixels, every one, and the same masks. The installed odcombine never writes
an output -- the template it lists its temporary spectra with matches nothing in
2.18's image lists -- so the gabarito is its own source with that one line corrected
(tests/data/iraf_odcombine/build.sh).

Its ways are kept, defects too. Every spectrum is combined from a copy of the first
spectrum's image, so the header values the combining reads are the first's for all:
`--scale exposure` scales nothing, a `!KEYWORD` scale, zero or weight is one value
for every spectrum, and so is a keyword's rdnoise or gain. A mask goes only to the
last spectrum taken for each output as each image is read: with `--group all` or
`images`, only an image's last aperture is masked. A mask with nothing bad left on
the output is no mask, though the log still names it. And the log's `statsec =` line
prints the date, where it means to print the section.
