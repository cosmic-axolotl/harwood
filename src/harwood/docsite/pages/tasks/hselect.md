`hselect` is how a night gets sorted: an expression over each header picks
the frames, and the keywords you name are printed for each one that matches.
`$I` is the image's own name, as in IRAF.

The expression is written as IRAF writes it: keywords in any case
(`imagetyp` is `IMAGETYP`), `==`, `!=`, `<`, `>`, and `&&`, `||` and `!` for
and, or and not -- Python's `and`, `or` and `not` work as well. `yes` (the
default) selects everything.

```example
title: The long exposures
run: harwood hselect @obj.lis @flat.lis '$I,IMAGETYP,EXPTIME' --expr "exptime > 10"
output: |
  obj003.fits
  matched: 3
```

```example
title: The flats, by their type
run: harwood hselect '*.fits' '$I,EXPTIME' --expr "imagetyp == 'flat' && exptime > 0"
output: |
  flat001.fits
  matched: 3
```

A keyword that no image has makes the expression false everywhere it is
used; rather than quietly matching nothing, hselect says which keyword it
could not find. IRAF's pattern match `?=` is not there: for *contains*,
write `'M92' in object`.

```warning
The expression only ever sees the header's own keywords. It is a filter,
not a place to run code: Python's built-in functions are not available in
it.
```
