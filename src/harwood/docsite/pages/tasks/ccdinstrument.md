IRAF's `ccdinstrument`: what a translation file makes of an image's header. For each parameter
the CCDRED tasks ask for, the keyword it reads, the default, the value the task would use and
what is in the image. A `?` is a value the header does not have.

```example
title: The table, for the frames of a night
after: night
run: harwood ccdinstrument obj001.fits --no-edit --parameters common
output: |
  imagetyp
```

`--parameters` is `basic`, `common` or `all`. Without `--no-edit` it takes commands, one to a
line, as in IRAF: `?` lists them. The name of a parameter, alone or with a keyword and a default,
changes its translation: `exptime ITIME`, `flatcor FLATCOR no`. `translate flat` says what the
value of the image type in this header means. `show`, `imheader`, `instrument`, `newimage`,
`next`, `read`, `write` and `quit` do what they say. On leaving, if anything changed it asks
whether to save it in the file.

Showing a subset adds it to `--ssfile`, `subsets` by default, as IRAF does. The other harwood
tasks do not read that file; they read the subset from the translation itself.

Two differences. `read` of the file in use goes back to what is in it, as it means to; IRAF
frees the table and is left with none. `translate` on an image with no image type says so and
goes on; IRAF ends there, and the changes so far are lost.
