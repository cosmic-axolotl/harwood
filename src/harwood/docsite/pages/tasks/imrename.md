IRAF's `imrename`: rename images, or move them into a directory.

```example
title: Give a processed frame a name that says what it is
after: reduced
run: harwood imrename red/obj001.fits red/field1_V_001.fits
output: |
  red/field1_V_001.fits
```

The new names are one per image (`a.fits,b.fits`, or an `@list`), or a
directory that already exists, into which the images move keeping their
names: `harwood imrename 'red/obj00*.fits' night1` after `mkdir night1`.

As in IRAF, an existing image is never overwritten (renaming an image to
itself is allowed). Unlike IRAF, which warns and goes on, harwood checks
every name first and renames nothing when one of them would overwrite
another: a night half renamed is harder to sort out than one not renamed at
all. Three names for one new name is refused the same way, with what to
write instead.
