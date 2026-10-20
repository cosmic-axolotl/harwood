Sets, adds or deletes header keywords, on one image or a whole set at once
-- a keyword edit is nearly always wanted on all the flats, or all of a
night, and doing it a file at a time is how half a set ends up different
from the other half. It is IRAF's `hedit`, ported from `hedit.x` with its
expression evaluator, and checked against IRAF 2.18 on 52 edits: the same
lines printed, the same cards written.

```example
title: Write the observer into every object frame
run: harwood hedit @obj.lis OBSERVER --value "A. Observer" --add
output: |
  add obj001.fits,OBSERVER = "A. Observer"
```

```console
$ harwood hedit '*.fits' DISPAXIS --value 2 --add
$ harwood hedit @obj.lis EXPTIME --value '(EXPTIME / 1000.)'
$ harwood hedit @obj.lis OBJECT --value '(OBJECT // " B")'
$ harwood hedit obj001.fits 'CCD*,GAIN' --value .
```

## The value

| `--value` | |
|---|---|
| `20`, `NGC 6341` | taken as it is |
| `(EXPTIME * 2)` | in parentheses: an expression, worked out for each image |
| `.` | prints the current value instead: `obj001.fits,EXPTIME = 60.` |

An expression is IRAF's: keywords of the header (in any case), numbers and
quoted strings, `+ - * / **`, `//` to join strings, comparisons, `&& || !`,
`?=` (pattern match), `cond ? a : b`, and the functions `abs acos asin atan
atan2 bool cos exp int log log10 max min mod nint real sin sqrt str tan`.
`$` is the keyword being edited, `$I` the image's name, `$F` the keyword's
name. Its rules are IRAF's too, and they matter: `7 / 2` is 3 (integers stay
integers), reals are single precision (7 digits), angles are in degrees, and
a keyword with `-` in it is written `@"DATE-OBS"`.

## What is written

Editing a keyword keeps its kind: a string stays a string, and a number is
written as the text of the new value (`5.`, `1.334567`, as IRAF prints
them). Adding one (`--add`) takes the kind of the value: a number becomes a
number, an expression's value has the expression's type (`(yes)` is a
logical T), and anything else is a string -- the word `yes` included, as in
IRAF. Quotes around the whole value keep digits as text: `--value "'4211'"`.

Where IRAF would write text without quotes into a numeric or logical card
-- `abc` into EXPTIME, `yes` into a T/F keyword -- the card it leaves is not
valid FITS, and other programs reject it. harwood writes the valid card
instead: `yes` and `no` into a logical keyword are T and F, other text is a
string.

## Choosing and checking

- `--fields` takes several names, and `*` and `?` match in any case:
  `'CCD*'`. A name that is not there is left alone and reported: a typo
  would otherwise create a new keyword and leave the real one as it was.
- `--add` adds the keyword where it is missing and edits it where it is
  there; `--addonly` adds it only where it is missing.
- `--delete` removes it.
- `--verify` asks before each change, as IRAF does by default: Enter
  accepts, `n` skips, anything else is the value to use; then
  `update image ? (yes):` decides whether that image is written.
- `--no-show` prints nothing; `--no-update` shows what would change and
  changes nothing.

What each keyword said before goes into the header's HISTORY: headers are
edited precisely when something was wrong, which is when the old value
matters.
