Marks the objects of a coordinate list on a displayed frame -- what was
found, what will be measured -- with IRAF's marks, numbers and labels, from
the list, the cursor, or a file of cursor commands; and keeps a snapshot of
the marked frame.

```console
$ harwood display obj001.fits
$ harwood tvmark 1 obj001.coo --image obj001.fits --mark circle --radii 5,10,20 --color 204
$ harwood tvmark 1 obj001.coo --image obj001.fits --number --outimage obj001.snap.fits
$ harwood tvmark 1 obj001.coo --image obj001.fits --interactive --logfile marks.log
```

The list is IRAF's: x and y in columns 1 and 2 and an optional label in
column 3, so a daofind `.coo` works as it is. `--image` displays the image
in the frame first; without it the frame must already hold one in the same
session (`display` at the window's prompt).

## Marks

| `--mark` | |
|---|---|
| `point` | a square dot `--pointsize` pixels wide (rounded up to odd) |
| `circle` | concentric circles of `--radii` (`5,10,20`: aperture and sky annulus) |
| `rectangle` | concentric rectangles of `--lengths` (`10,20 0.5`: lengths, then width/length) |
| `plus`, `cross` | + and x, `--txsize` font cells across |
| `none` | nothing -- with `--number` or `--label`, only the text |

`--number` writes each object's number in the list and `--label` its
column 3, with the lower left corner of the text at the object, moved by
`--nxoffset` and `--nyoffset`. `--color` takes IMTOOL's numbers, as IRAF
does: 202 black, 203 white, 204 red, 205 green, 206 blue, 207 yellow, 208
cyan, 209 magenta, 210 coral, 211 maroon, 212 orange, 213 khaki, 214
orchid, 215 turquoise, 216 violet, 217 wheat; 255, the default, draws in the
window's own mark colour.

## With the cursor

`--interactive` hands the window's keys to tvmark until `q`:

| key | |
|---|---|
| `+` `x` `.` | a plus, a cross, a dot at the cursor |
| `c` `r` | the circles (`radii`) or rectangles (`lengths`) at the cursor |
| `s` `v` `b` `f` | press twice: a line, a circle (centre, then edge), a rectangle, a filled one |
| `e` | press twice: the region goes back to what `:save` kept |
| `n` `p` `l` | mark the next, the previous, or every object of the list |
| `m` `-` `o` | move to the next or previous object, or back to the first |
| `a` | add the cursor position to the list, and mark it |
| `d` | delete the object nearest the cursor (within `--tolerance`) |
| `k` | keep the last command in the logfile |
| `?` `q` | the keys; the end |

Colon commands, typed at the prompt: `:show`, `:move N`, `:next N M`,
`:text words`, `:save`, `:restore`, `:write [name]`, and every parameter --
`:mark circle`, `:radii 4,8`, `:color 204`, `:number yes`.

Objects added with `a` are appended to the coordinate file; deleted ones
are taken out of it, every other line (a daofind header, the other columns)
left as it was, and with `--deletions ext` they are written to
`coords.ext`.

## Logs and command files

`--logfile` keeps the commands -- each one with `--autolog`, otherwise those
kept with `k` -- as `x y frame key` lines, which is also what `--commands`
reads: a session can be redone, or written by hand, without the cursor.
With neither the cursor nor a command file, every object of the list gets
the default mark.

## The snapshot

`--outimage` writes the marked frame after marking the whole list; with the
cursor or a command file, `:write [name]` takes one whenever it is given
(`outimage.snap.N`, or the image's name `.snap.N`, without a name). A FITS
snapshot holds the frame's grey levels, 1 to 200, as [display](task:display)
computes them, with the marks at their colour number -- IRAF's snap -- so
`harwood display snap.fits --ztrans none` shows it as it was; a `.png`
snapshot is the picture, the marks in their colours.

The marks are drawn by harwood, not ported from IRAF's frame-buffer
graphics: they sit at the list's positions with IRAF's sizes, but the
pixels a circle or a letter covers in the snapshot have not been compared
with IRAF's, and need not be the same ones.
