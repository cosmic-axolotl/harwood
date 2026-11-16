The illumination along the slit of a long-slit sky flat, as IRAF's
`longslit.illumination` makes it: the image averaged along the dispersion in
bins, the profile across the slit of each bin fitted with ICFIT, normalized,
and interpolated between the bins over the whole image. Divided into a frame,
it flattens the sky along the slit, which a dome flat does not light as the
sky does.

```console
$ harwood illumination SkyFlat.fits Illum.fits --dispaxis 2 --function chebyshev --order 2 --high-reject 1 --grow 2 --niterate 2
$ harwood illumination flat_response.fits flat_illum.fits --bins "1:222 109:220 330:440" --interactive
$ harwood imarith Resp.fits '*' Illum.fits SuperFlat.fits
```

The bins are `--bins`, ranges along the dispersion, or `--nbins` equal ones
(five by default) -- cut as IRAF cuts them, npts/nbins pixels each in
integers, so the pixels left over at the end are in none. Ranges are clipped to
the image and sorted, and may overlap. Each bin is fitted in turn, with the
fit's parameters carrying on to the next as IRAF's do.

By hand (`--interactive`) the task goes as IRAF's does. First the bins' graph: the
image averaged across the slit, along the dispersion, with each bin marked by a bar
between two ticks (and shaded, so that bins which overlap are seen to). Its keys:

| key | what it does |
|-----|--------------|
| `s` | a bin from here to where the next key is pressed, added to those there are |
| `i` | clear the bins: all the dispersion is one bin, until `s` marks others |
| `:bins 1:200 400:600` | set them; `:bins` and `:show` print them |
| `:/xwindow 1 300` ... | gtools' commands |
| `q` | done |

Then, before each bin, the window asks whether to fit it by hand -- IRAF's four
answers: yes and no for this bin, "Yes to all" and "No to all" for the rest -- and a
yes opens the icfit window on the bin's profile. With several images the question is
asked for each image first; a no there skips the bins' graph, not the questions about
its bins.

Without a window, `--cursor` reads the keys from a file in IRAF's cursor format
(`x y 1 key`, or a bare key or `:command`): the bins' graph up to its `q`, then each
fit made by hand up to its own; and `--answers` the replies to "Determine
illumination interactively for <image>" and "... at bin <n>" (no, yes, NO, YES), a
line each -- yes to each without it:

```console
$ cat bins.cur
0 0 1 i
50 500 1 s
10 500 1 s
70 500 1 s
140 500 1 s
0 0 1 :show
0 0 1 q
$ printf 'yes\nNO\n' > answers.txt
$ harwood illumination sky2 o.fits --cursor bins.cur --answers answers.txt
```

What IRAF does there, kept: a bin marked with `s` is added to the bins there are --
the five default ones too, which it then overlaps -- unless `i` cleared them first;
the positions are cut to integers (60.6 is 60); a bin marked right to left, or past
the end of the image, is taken between its ends and cut to the image; `s` and then
`q` adds nothing; after one gtools command every colon command redraws the graph; a
colon command that is not one says nothing. One thing is not: IRAF labels the graph's
axis with the other axis' name -- "Column" under line numbers. harwood says which it is.

As IRAF does it, each profile is normalized at its pixel nx/2 - 1, not the
middle of the slit; a bin sits at (first + last)/2, in integers; between two
bins the image is mapped onto the interpolator's grid so that a bin's centre
and the pixel after it land on the same place. The interpolator is
`--interpolator`, poly3 by default, falling back to linear with fewer than four
bins (poly3, spline3) and to poly3 with fewer than six (poly5). With one bin,
its profile is copied over the whole image.

The output carries CCDMEAN = 1 and MKILLUM, which [ccdproc](task:ccdproc)'s
`--illumcor` requires; ccdproc computes its mean again, as IRAF's does.

On the course's 6.1 frames the illumination of SkyFlat with the text's fit is
IRAF's Illum.fits pixel for pixel, and Resp times Illum its SuperFlat; on 7.1,
the three bins she marked and the orders 8, 8 and 6 she chose give
flat_illum.fits, and ccdproc's flat and illumination correction with both
gives the galaxy and the lamp IRAF made. Every case compared on synthetic sky
flats -- one to seven bins, bins in any order or overlapping, the five
interpolators and their fallbacks, both dispersion axes, sections -- gives
IRAF 2.18's file pixel for pixel. By hand, nine sessions from cursor files and
answers -- bins added, cleared and marked afresh, set by `:bins`, on both axes and on
an axis run backwards; yes, no, YES and NO to images and bins -- ask and print what
IRAF does, draw its bins' graph, each mark where its metacode has it, and write its
illumination.
