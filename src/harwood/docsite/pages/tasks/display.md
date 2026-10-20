Loads an image into a frame of the window -- harwood's own, so there is no
ds9 to start -- with z1 and z2, the values shown as black and white, chosen
as IRAF's `display` chooses them, and printed as it prints them. With
`--coords`, each position of a list is circled, which is what `tvmark` is
mostly used for: seeing what daofind found and what it missed.

```console
$ harwood display red/obj001.fits --coords obj001.coo
$ harwood display red/obj002.fits --frame 2 --contrast 0.5
$ harwood display red/obj003.fits --no-zscale --no-zrange --z1 900 --z2 1400
```

## How z1 and z2 are chosen

The arithmetic is IRAF's, ported from `images/tv/display/zscale.x` and
checked against IRAF 2.18 on 99 images and settings: the same z1 and z2 to
every digit IRAF prints, and -- through a stand-in for the image server that
kept what IRAF sent it -- the same grey level in every pixel.

| | |
|---|---|
| `--zscale` (default) | sample about `--nsample` pixels on a regular grid, sort them, fit a straight line to the middle of the sorted values rejecting outliers, and spread its slope by `--contrast` around the median. Sky and faint stars show at once |
| `--contrast` | 0.25. Smaller widens the range (z2 - z1 grows); 0 gives the extremes of the sample |
| `--nsample` | 1000, at least 100. The sample is every i-th pixel of every i-th line, i chosen to give about this many |
| `--zmask` | sample these pixels instead: a mask, or a section such as `[100:400,*]` |
| `--bpmask` | pixels left out of the sample. `BPM` (the default) is the mask the image's BPM keyword names; no keyword, no mask |
| `--no-zscale` with `--zrange` | the extremes of the sample |
| `--no-zscale --no-zrange` | `--z1` and `--z2`, which must then be given |

Blank (NaN) pixels are left out of the sample as bad ones are -- IRAF's
images had none.

## From values to grey

| `--ztrans` | |
|---|---|
| `linear` | z1..z2 onto the grey scale |
| `log` | z1..z2 onto 1..1000, then the logarithm, onto the grey scale -- the stretch box's *log* |
| `none` | the values themselves as grey levels; the frame keeps 8 bits, so they wrap every 256 and show contours |
| `user` | a table, `--lutfile`: two columns, value and grey level (1 to 200) |

With `none` and `user`, IRAF sets z1 and z2 to the frame's grey range, 1 and
200, and harwood does the same. For `user` this means the table is applied
to the values 1 to 200, whatever range its first column covers: IRAF reads
that range and then overwrites it. A table written for IRAF gives the same
picture here.

In the window, the scale box reads *z1 z2* while it shows the limits the
task chose; picking *zscale*, *minmax* or *99.5%* there recomputes them, and
the stretch box changes the transfer, without touching the pixels.

## Masks

`--bpmask` and `--zmask` take the names every harwood task takes, as IRAF's
do: a FITS image or an IRAF `.pl` mask (read without IRAF), `BPM` or
`!KEYWORD` for a file a keyword names, a section, a text file of
`c1 c2 l1 l2` rectangles, `EMPTY`, and `^` before any of them to invert it.
A mask made for the whole detector is matched to a trimmed frame or a
section through LTV and LTM, so it follows the pixels.

## Frames and windows

Frames are numbered as in IRAF; the toolbar's *frame* picks one, *blink*
alternates between them and *tile* shows them side by side. From the
window's prompt, `display` loads into the window you are looking at; from a
terminal, it opens one, and `--no-wait` returns at once instead of waiting
for it to be closed.

Without a screen -- over ssh, in a script -- the picture is written to a
PNG instead, and the command says where. [impicture](task:impicture) writes
one on purpose.

Not ported: IRAF's display-window geometry (`xcenter`, `ycenter`, `xsize`,
`ysize`, `xmag`, `ymag`, `fill`, `order`), which the window's zoom and pan
replace, and overlay masks drawn in colour, which `--coords` and
[tvmark](task:tvmark) do.
