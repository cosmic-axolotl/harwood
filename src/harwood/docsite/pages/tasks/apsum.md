Extracts the spectra of a frame through apertures already defined, its own
or another frame's. It is [apall](task:apall) without recentering or
resizing, as `apsum` is in IRAF; apsum also traces the apertures again, as
IRAF's does, unless `--no-trace` says to use the stored trace. The
extraction itself -- plain sum or variance weights, background, cleaning --
is apall's, with the same options; see [its manual](task:apall).

```console
$ harwood aptrace obj.fits --nfind 2
$ harwood apsum obj.fits obj.ms.fits --no-trace --weights variance \
      --background fit --gain GAIN --readnoise RDNOISE
```

Through another frame's apertures, as for an arc:

```console
$ harwood apsum arc.fits arc.ms.fits --references obj --no-trace
```

The spectra go to one FITS file with a table per aperture, as apall writes
them. With no apertures stored, apsum finds them first.
