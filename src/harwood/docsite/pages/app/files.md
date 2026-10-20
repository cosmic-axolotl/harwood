# Text files, several at once

The Files panel holds text files in tabs: @lists, coordinate files, the
tables phot and daofind write, extinction tables. Open them with
*File › Open text files* (Ctrl+O), the 📂 button, or from the prompt:

```text
hw> edit obj001.coo obj001.mag
```

From a terminal, `harwood edit obj001.coo` opens them in a window of their
own.

The bar under the tabs reads the line the cursor is on for its columns:

```text
line 7 · column 3 of 7: MAG · iraf, 7 columns, 3 rows
```

-- which column a number is in being exactly what is hard to see in a table
that has lost its alignment. A tab with changes is marked **●** and its tip
says *unsaved changes*; closing it asks first. Ctrl+S saves the file in
front, Ctrl+Shift+S all of them.

## Edit together

**🧰 Edit together…** (Ctrl+Shift+T) makes one edit to several of the open
files:

| edit | |
|---|---|
| Find and replace | text, or a regular expression |
| Delete rows containing | rows of data with a given text in them |
| Delete columns, Keep only columns | by number: `3`, `2,4`, `2-5` |
| Insert a column | a value, and its name in the header if there is one |
| Set a column to a value | |
| Arithmetic on a column | `+ - * /` a number: shift every x by 3 pixels |
| Sort rows by a column | by number when the column holds numbers |

### The check, before anything is changed

Editing a batch at once is only safe when the files have the same shape.
"Delete column 4" is the magnitude in one file and its error in another if
the two were written by different versions of a task, and nothing about the
result would say so. So the dialog reads each file for its shape -- format,
columns, rows, the names of the columns -- and shows them side by side, and
under *What to know first* it lists every difference that matters to the
edit you are setting up:

- the files do not have the same number of columns;
- a column is a different quantity in another file (`mag` in one, `merr` in
  the other);
- some rows are shorter than the rest;
- a column the edit needs is not there in some file;
- text where arithmetic wants numbers;
- a file writes each star over several lines, as phot's text output does;
- a file is not a table at all.

With nothing to say it says **✔ ready**, and the edit is made. With
something, applying asks **Are you sure?**, listing each reason with what to
do about it; **Go back** is the default, **Apply anyway** goes ahead. Going
ahead anyway is written in the session, so it is on record.

Either way the edit goes into the tabs and not to disk: read it, undo it
with one Ctrl+Z per file, and only Save writes it.

### What is kept

- harwood's own tables (ECSV) are edited through astropy, so their header
  still declares exactly the columns there are; an edit to the text alone
  would leave the header describing a column that is gone, and the file
  would no longer read;
- IRAF's `#N`, `#U` and `#F` header lines are edited with the columns they
  describe;
- the alignment, the number of decimals, `INDEF`, the comments and the line
  endings stay as they were;
- a file whose records continue over several lines is left alone by column
  edits, and the dialog says so.

## The same check from a terminal

```example
title: Compare two tables before editing them together
after: found
run: harwood edit --check obj001.coo bright.coo
output: |
  the files do not have the same number of columns
```
