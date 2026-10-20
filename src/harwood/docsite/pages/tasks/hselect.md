`hselect` is how a night gets sorted: an expression over each header picks
the frames, and the keywords you name are printed for each one that matches,
one line per image with a tab between values -- the lines that go into an
`@list`. `$I` is the image's own name, as in IRAF.

```example
title: The long exposures
run: harwood hselect @obj.lis @flat.lis '$I,IMAGETYP,EXPTIME' --expr "exptime > 10"
output: |
  obj003.fits
```

```example
title: The flats, by their type
run: harwood hselect '*.fits' '$I,EXPTIME' --expr "imagetyp == 'flat' && exptime > 0"
output: |
  flat001.fits
```

```console
$ harwood hselect '*.fits' '$I' --expr "imagetyp == 'object'" > obj.lis
$ harwood hselect @obj.lis '$I,OBJECT' --expr "object ?= 'M92*'"
```

The expression is evaluated as IRAF evaluates it -- the same evaluator as
[hedit](task:hedit)'s, ported from IRAF: keywords in any case (`imagetyp` is
`IMAGETYP`), `==`, `!=`, `<`, `>`, `&&`, `||` and `!`, `?=` for a pattern
(`'M92*'`, `'^NGC'`), arithmetic (`exptime / 60 > 2`) and the functions.
`yes` (the default) selects everything. Python's `and`, `or` and `not` are
read as `&&`, `||` and `!`.

Values are printed as the cards hold them (`60.`, `T`), a string with blanks
in double quotes; a field the header does not have is left out of the line,
as IRAF does. `*` in `fields` matches: `'$I,CCD*'`.

A frame without a keyword the expression uses is not selected, and hselect
says which keyword it could not find -- where IRAF stops the whole listing
at the first such frame.
