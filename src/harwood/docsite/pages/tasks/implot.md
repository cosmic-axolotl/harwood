Plots lines and columns of an image in a panel of the window, as IRAF's
`implot`. It is how sections get decided: you look at where the
illuminated pixels stop and the overscan begins, rather than trusting that
the header is right about it.

```console
$ harwood implot obj001.fits
$ harwood implot obj001.fits --column 110
$ harwood implot 'obj00*.fits' --nsum 5
```

The keys and colon commands are IRAF's, answered by a port of
`t_implot.x`; checked against IRAF 2.18 with a cursor file, it prints the
same lines.

| key | |
|---|---|
| `l`, `c` | plot the line or column under the cursor -- in a line plot, the line is read off the right-hand axis, which spans the plot's height from line 1 to the last |
| `a` | press at both ends of a range: the average of those lines or columns |
| `e` | expand: two corners of the part to see (only in x if the two are level) |
| `j`, `k` | move down or up by `--step` lines (or columns), replacing the curve |
| `m`, `n` | the previous or next image of the list, same line or column |
| `o` | overplot the next one |
| `p` | profile: two points mark the region and a straight background; the peak, its width at half height, and the moments (centroid, flux, width, asymmetry) |
| `s` | statistics between two points: median, mean, rms, sum, npix |
| space | the pixel under the cursor and its value |
| `r` | redraw |
| `q` | done |

| command | |
|---|---|
| `:l 20` `:l 20 30`, `:c 105` `:c 105 120` | a line or column, or the average of a range |
| `:a 5` | how many l and c average |
| `:i image` | another image |
| `:o` | overplot the next |
| `:log+`, `:log-` | logarithmic counts, from the next redraw (`r`), with the axis out to whole decades, as IRAF has it |
| `:step 7` | the step of j and k |
| `:w logical` / `physical` / `world` | the coordinates along the plot |
| `:x 100 200`, `:y 400 900` | fix the range (no numbers: free again) |

Without the window, `--coords` reads the keys from a file, IRAF's graphics
cursor form -- `x y 1 key` or `x y 1 :command`, `\040` for the space bar --
and prints what they print:

```console
$ harwood implot obj001.fits --coords cur
pixel=[150,200] value=1324.802
```

On a raw frame of the [sample night](page:start/sample), a line shows about
800 ADU of sky and stars over columns 1 to 100 and the 500 ADU of the
overscan over 105 to 120 -- the strip ccdproc fits and subtracts.

The right-hand axis counts the image's lines or columns, and the + on it is where
the plotted one lies. Seven cursor files IRAF ran give the same text and, from its
graphics metacode, the same plots: windows, labels, curves, marks and the dashed
lines of `p`. `p` prints its four lines at once, where IRAF's `/` walks through them.

Contours and surfaces are IRAF's `contour` and `surface`, separate tasks.
