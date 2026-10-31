The same as [mknobsfile](task:mknobsfile), for a field of many stars, where it is
the image sets that come last on IRAF's command line: they are given here with
`--imsets`.

```console
$ harwood mkobsfile @all.lis B,V observations --imsets imsets.mkimsets --tolerance 8
```

In IRAF the two differ only in what they ask: left to itself, mkobsfile takes the
image sets, the observing parameters, the shifts and the aperture corrections at the
terminal, one at a time, through a task of its own, mkphotcors. That is not ported,
because a question at the terminal cannot be repeated or checked afterwards: the
sets come from [mkimsets](task:mkimsets), and the others from their files, as options.

**Identical to IRAF.** The observations file, the format file and what it prints
agree byte for byte with IRAF's on the twelve Skinakas images, with each option. One
difference: when `--obsparams` is given IRAF's mkphotcors warns that the filter in
each line is an image not in the sets, having been told the wrong column, and goes
on; that warning does not exist here. The file written is the same.
