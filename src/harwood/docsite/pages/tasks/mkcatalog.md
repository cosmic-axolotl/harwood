Makes a catalog of standard stars -- the magnitudes and colours
[fitparams](task:fitparams) fits the night's observations to -- in IRAF's
mkcatalog format, or adds stars to one, and writes the format file
[mkconfig](task:mkconfig) builds the configuration's catalog section from.

```console
$ harwood mkcatalog standards --columns "ID,V,error(V),BV,error(BV)" --records stars.txt
```

**The columns.** The first is the star's id, which the observations are
matched by; the rest are named as the configuration will use them -- a
letter then letters and digits, or `error(V)` for the error of `V`. Widths
are 15 for the id and 10 for the others unless `--widths` gives them; a
value longer than its column is cut, as IRAF cuts it, and listed.

**The records.** `--records` is a text file, one star per line, its values
separated by blanks; values missing at the end are `INDEF`, lines starting
with `#` are skipped and a blank line stays a blank line. The catalog's
header says what its columns are, so the same command without `--columns`
adds more stars to it later.

**Files.** `standards` holds the header (title, number of columns, the
record length, the names and widths) and the records; `fstandards.dat` is
its catalog section, for `harwood mkconfig cfg standards ...`. A catalog
does not have to be made this way: [fitparams](task:fitparams) reads any
text file with the id first and the columns the configuration names.

**Identical to IRAF.** Header, records and format file are IRAF's, byte for
byte. IRAF types the columns and records at prompts; here they are given
at once, and the review of old records is not ported -- edit the file. A
catalog of more than ten columns is where IRAF loses the eleventh column's
name, through a pointer kept across a reallocation; harwood writes it.
