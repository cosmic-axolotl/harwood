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

**Typed, as IRAF asks.** With neither `--columns` nor `--records`, mkcatalog
asks as IRAF's does: for a new catalog the name and width of each column, a
blank name ending them, then the stars one value at a time, a blank id ending
them; for an old one only the stars, its columns read from its header.
`--verify` asks again for each column and value, to change it; `--review`
shows the old records value by value first, to keep or change; `--edit` opens
the catalog in the editor at the end -- a dialog in the window, `$EDITOR` at a
terminal. `--answers` reads the lines from a file instead of the keyboard. A
record cut short by the end of the input is not written, as in IRAF; a
catalog whose title is not its name, or a file mkcatalog did not make, is
refused with IRAF's message. IRAF has the editor on; harwood has it off, so
that a run can be repeated.

**Files.** `standards` holds the header (title, number of columns, the
record length, the names and widths) and the records; `fstandards.dat` is
its catalog section, for `harwood mkconfig cfg standards ...`. A catalog
does not have to be made this way: [fitparams](task:fitparams) reads any
text file with the id first and the columns the configuration names.

**Identical to IRAF.** Header, records and format file are IRAF's, byte for
byte, given at once or typed; typed, what is asked and said is IRAF's too, on
six sessions (new catalog, verify, records added, review, the editor) and the
two refusals. A catalog of more than ten columns is where IRAF loses the
eleventh column's name, through a pointer kept across a reallocation;
harwood writes it.
