IRAF's `mskexpr`: a pixel mask from an expression, evaluated a line at a time with the
same types and functions as [imexpr](task:imexpr), and with regions.

```example
title: Bad values of a frame, and a circle round a bright star
after: reduced
run: >-
  harwood mskexpr "(i < 0 || i > 60000 || circle (100., 120., 8.)) ? 1 : 0" bad.pl --refimages red/obj001.fits
output: |
  Creating mask bad.pl using reference image red/obj001.fits
```

`--refimages` gives each mask its size and header and the operand `i`, its keywords as
`i.EXPTIME`; `--refmasks` the operand `m`, cut or padded to the image's size. With neither,
`--dims` gives the size. `I`, `J`, `K` are the coordinates of the pixel. A mask that
already exists is left as it is.

| region | arguments |
|---|---|
| `point` | `x, y` |
| `circle` | `xc, yc, r` |
| `ellipse`, `rectangle` | `xc, yc, r, ratio, theta` |
| `box` | `x1, y1, x2, y2` |
| `vector` | `x1, y1, x2, y2, width` |
| `pie` | `xc, yc, theta1, theta2`, as far as the mask's edge |
| `polygon` | `x1, y1, ..., xn, yn` |
| `cannulus` | `xc, yc, r1, r2` |
| `eannulus`, `rannulus` | `xc, yc, r1, r2, ratio, theta` |
| `pannulus` | `width, x1, y1, ..., xn, yn` |
| `cols`, `lines` | a range, as a string: `"100-200,300"` |

Each also takes two coordinates first, `circle (I, J, xc, yc, r)`, or `circle (J, K, ...)` for
another plane of a cube. A pixel is inside by its centre, with angles in degrees.

The expression has to come to integers: a real is refused. `--depth` keeps that many bits
of the values. A constant gives an empty mask, because IRAF's way of writing one clears
the line; say `repl (5, 512)` for a line of 5.

Each region was run in IRAF 2.18, and these are its bytes. A polygon can have more than
IRAF's eight vertices, where its 17 arguments to a function stop it.
