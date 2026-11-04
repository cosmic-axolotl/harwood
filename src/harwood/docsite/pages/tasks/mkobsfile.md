The same as [mknobsfile](task:mknobsfile), for a field of many stars, where it is
the image sets that come last on IRAF's command line: they are given here with
`--imsets`.

```console
$ harwood mkobsfile @all.lis B,V observations --imsets imsets.mkimsets --tolerance 8
```

In IRAF the two differ only in what they ask: mkobsfile first runs
[mkphotcors](task:mkphotcors), which takes the image sets, the observing parameters,
the shifts and the aperture corrections from their files, or, for those given as
`STDIN`, asks for them at the terminal, one image at a time. The image sets are
`STDIN` unless `--imsets` names a file, as in IRAF; the others are only asked for
when given as `STDIN` (in IRAF that is their default too). `--verify` asks again for
each answer, `--answers` reads them from a file. The sets are usually better made
by [mkimsets](task:mkimsets), which can be checked afterwards.

**Identical to IRAF.** The observations file, the format file and what it prints
agree byte for byte with IRAF's on the twelve Skinakas images, with each option and
with the tables typed. When `--obsparams` is a file, IRAF's mkphotcors warns that the
filter in each line is an image not in the sets -- it is given the columns without
the image's -- and goes on; harwood says it too. The file written is the same.
