Asks for what [mkobsfile](task:mkobsfile) needs and the photometry does not say:
which images make each set, and, for each image, its filter, exposure time, airmass
and time where the photometry has them wrong, the shift in x and y that lines its
stars up with the rest of its set, and its aperture correction. Each of the four is
a file: one that exists is read, one that does not is asked for at the terminal,
image by image, and written, so that the next run reads it; the image sets left
empty are asked for and not written. mkobsfile runs it first.

```console
$ harwood mkphotcors sets B,V obsparams shifts apercors
```

```console
Enter name of image set 1 (name, <EOF>=quit entry): PG2331
    Enter image name 1 (name, <CR>=INDEF): stan0051cr.fits
    Enter image name 2 (name, <CR>=INDEF): stan0052cr.fits
Enter name of image set 2 (name, <EOF>=quit entry):
```

The questions are IRAF's, answered as IRAF's are: a blank line is the default the
prompt shows, the end of the input (Ctrl-D) ends the table. `--verify` asks again
for each answer, to change it; `--answers` reads what would be typed from a file;
in the window each question is a dialog. The files are written as IRAF writes them,
and the sessions say and write what IRAF's do.
