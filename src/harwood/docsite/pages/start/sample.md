# The sample night

Every example in this documentation runs on the same small night of data,
which harwood writes for you:

```console
$ harwood docs --sample night
$ cd night
```

It is made on the spot, the same bytes every time, so the numbers on these
pages are the numbers you will get. (The test suite runs every example on
it and checks.)

| file | what it is |
|---|---|
| `bias001.fits` … `bias003.fits` | zero frames: a 500 ADU level and 5 ADU of read noise |
| `flat001.fits` … `flat003.fits` | V dome flats, with vignetting and a pixel-to-pixel pattern |
| `obj001.fits` … `obj003.fits` | a V field of thirty stars (FWHM 3 pixels) on 300 ADU of sky; the telescope drifts a few pixels between them, and obj003 was hit by eight cosmic rays |
| `bias.lis`, `flat.lis`, `obj.lis` | the IRAF lists of each, for `@bias.lis` |
| `bright.coo` | three bright stars on obj001, as `x y` (1-based) |
| `stars.cur` | a cursor file for imexamine: `r` and `a` on the two brightest stars |

Every frame is 120 × 100 pixels: the light falls on `[1:100,1:100]`
(`TRIMSEC`), and columns 105 to 120 are an overscan strip (`BIASSEC`), as
on a CCD of the 1980s. The headers say what each frame is (`IMAGETYP`), the
filter (`FILTERS`), the exposure, the gain and read noise, the time and the
pointing.

## Where an example starts

Each example says where it starts from: the night as written, or the night
part of the way through [its reduction](page:start/first-night). Each state
is the one before it plus these commands, run in the night's directory
(every example also has them, under *How to get there*):

```generated
stages
```

```example
title: The sample night, listed
run: harwood ccdlist @bias.lis @flat.lis @obj.lis
output: |
  bias001.fits
  flat001.fits
```
