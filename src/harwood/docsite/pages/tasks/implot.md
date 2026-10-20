Plots lines and columns of an image in a panel of the window, as IRAF's
`implot`. It is how sections get decided: you look at where the
illuminated pixels stop and the overscan begins, rather than trusting that
the header is right about it.

```console
$ harwood implot obj001.fits
$ harwood implot obj001.fits --column 110
```

The keys are the ones the tutorials tell you to press:

| key | |
|---|---|
| `l`, `c` | plot the line or column under the cursor |
| `:l 20 30`, `:c 105 120` | average a range of lines or columns |
| `e` | expand: mark two corners of the part to see |
| `r` | redraw |
| `C` | the cursor position and value |
| `o` | overplot the next one |
| `q` | done |

On a raw frame of the [sample night](page:start/sample), a line shows about
800 ADU of sky and stars over columns 1 to 100 and the 500 ADU of the
overscan over 105 to 120 -- the strip ccdproc fits and subtracts.
