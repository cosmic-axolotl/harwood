Opens the spatial profile of a frame with its apertures drawn on it, to set
them by hand with IRAF's apedit keys, and writes what you leave to
`database/ap<image>`. It is [apall](task:apall) with only `--edit` on and
nothing found, as `apedit` is in IRAF: the apertures are the ones stored for
the frame, or for `--references`, and a frame with none starts empty, for
`m` or `f` to make them.

```console
$ harwood apedit obj.fits
```

The keys are apall's; see [its manual](task:apall). `--gcommands` replays a
file of IRAF cursor commands instead of opening the window.
