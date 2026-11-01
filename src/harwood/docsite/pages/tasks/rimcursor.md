IRAF's `rimcursor`: each read of the cursor as a line of a list, `x y wcs key`, the position in
the system asked for. What it prints is what `imexamine` and the others take as a cursor file.

A file of `reads.txt` with the lines `10 20 101 a` and `33.5 44 101 b`, in physical coordinates:

    $ harwood rimcursor red/obj001.fits --wcs physical --cursor reads.txt
    10. 20. 101 a
    33.5 44. 101 b

Without `--cursor` the image opens in the window: a key is a read, and a colon a command, and
closing the window ends the list. With `--cursor` the reads come from a file of `x y wcs key`,
or from the terminal with `STDIN`. `--wxformat` and `--wyformat` give the formats; with neither
it is the format in the image's WCS or else `%0.15g`.

The lines are IRAF's, compared on 15 cases of coordinates and formats, but for a `dec` of a TAN
image that in one read comes out a unit lower in the 15th digit.
