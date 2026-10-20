Writes an image as a picture, as the window shows it: the same limits,
stretch and colours give the same pixels as *File › Export image*. The
whole frame at its own resolution, with north as in the FITS file (row 1 at
the bottom).

```example
title: A picture of the processed frame
after: reduced
run: harwood impicture red/obj001.fits obj001.png --stretch asinh
output: |
  -> obj001.png
```

| | |
|---|---|
| `--scale` | how the limits are chosen: `zscale` (IRAF's), `minmax`, `99.5%` |
| `--z1`, `--z2` | the limits themselves, overriding the scale |
| `--stretch` | `linear`, `sqrt`, `log`, `asinh` |
| `--colormap` | `grey`, `viridis`, `cividis`, `magma`, `heat`, `cool`, `sepia` |

The format comes from the name: PNG needs nothing installed; JPEG, TIFF and
WebP need the window's extra, `harwood[gui]`. Several images and a template
for the names write a whole night at once.

This is not IRAF's `export`, which writes raw and old raster formats with
parameters of its own: a different task, so a different name.
