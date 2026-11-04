`identify` is where a spectrum stops being counts against pixels and
becomes counts against wavelength: the lines of an arc lamp are found,
each is matched to a laboratory wavelength from the lamp's line list
(`--coordlist`, IRAF's `linelists$` lists included), and a smooth function
from pixel to wavelength is fitted through them. The fit is IRAF's icfit --
`spline3` of one piece, a cubic, by default -- and the solution goes to
`database/id<image>`, where [refspectra](task:refspectra) and
[dispcor](task:dispcor) look for it.

A two-dimensional arc is collapsed first, `--nsum` lines or columns across
the slit (the lamp fills the slit, so this costs nothing and buys signal),
so it can be given straight from [ccdproc](task:ccdproc).

## Two ways to say which line is which

**By hand, as IRAF asks.** `--interactive` opens the arc in a panel: mark
four or five lines spread across it with `m` and type each wavelength, `f`
fits, handing the keys to icfit until its `q` (`d` there deletes the feature), `l`
adds every other catalogue line the fit explains, `d` deletes a bad one, `q` asks
whether to write them. Spread matters more than number: five lines bunched in
the middle constrain the ends not at all.

The rest of IRAF's keys are there, along the panel's foot: `e` and `y` the list or the
peaks with no fit, `c` and `a c` centre again, `u t v` a feature's coordinate, place
and weight, `. + n -` move between features, `g s x` shift them, `b` identifies
automatically, `i` starts again, `z` and `p` zoom and pan. So are the colon commands
for `match`, `minsep`, `fwidth` and the like, and `:features` and `:show`. Marking a line
within `minsep` of a feature takes its coordinate from it: type it again. Thirteen
sessions IRAF ran say and write the same and draw the same, from its graphics metacode.

On a two-dimensional arc or an extracted spectrum of several apertures, `j` and `k`
go to the previous and next line -- `nsum` lines on, round the end of the frame --
or aperture, and `o` to the one you type. Each line left is kept and comes back as
it was; a line with a record in the database starts from it, one without from the
features in hand, which the solution of a neighbour usually needs only `a c` and
`f` to fit; `q` writes every line changed. `:read` and `:add` take the features of
another line's record (`:read arc 3`) or another image's, in place of these or among
them, `:write` writes this one at once, and `:nsum` sets how many lines are summed.
A session starts from the record the line already has, as IRAF's does. Five more
sessions IRAF ran say and write the same. Not ported: `w` and the `/` commands (the
mouse zooms), and `:image`, which goes on to another image: run identify on it.
harwood does not write `REFSPEC1` into the arc's header, as IRAF does with every
record: [refspectra](task:refspectra) says which arc a spectrum takes.

```console
$ harwood identify p/sp0018.fits --coordlist 'linelists$thar.dat' --section 'middle column' --interactive
```

**Automatically, from where the spectrum sits.** Give the wavelength at the
middle and the dispersion, good to about one per cent, and IRAF's
autoidentify places the lines itself:

```console
$ harwood identify arc.fits --coordlist 'linelists$henear.dat' --crval 6300 --cdelt 2.5
```

With neither, `identify` stops at once and says so, with these two ways
forward. The search over every dispersion a list allows exists -- it is
`autoidentify arc.fits INDEF INDEF`, as in IRAF -- but against a dense list
it is slow and rarely conclusive: on the 1992 thorium arc it ran past half
an hour without an answer. Against a thorium-argon list, with a line every
couple of Ångströms, several quite different dispersions explain a couple
of dozen lines each, and a cubic fits any of them to a tenth of a pixel. So
an answer is checked before it is believed, and when no candidate explains
enough of both the arc and the list, harwood says there is no solution
rather than write one: see
[where the 1992 night stops](page:recipes/1992-spectroscopic-night#7-where-it-stops).

**Which way the spectrum runs.** On a two-dimensional arc, `--section`
says which line or column to read: `middle line` (the default, as in IRAF)
or `middle column`, or an explicit `[51,*]`. A section across the slit
instead of along the spectrum holds a handful of pixels and no lines;
harwood says so, and suggests the other axis, rather than search it.

## How good it is

An arc line's centre is measured to about a tenth of a pixel, and that is
the yardstick: the fit's rms is reported in Ångströms and in pixels, and a
solution that scatters by more than a third of a pixel gets a warning that
something has been matched to the wrong catalogue entry. Rejected and
deleted lines are kept in the database: which lines were thrown out is part
of the solution.
