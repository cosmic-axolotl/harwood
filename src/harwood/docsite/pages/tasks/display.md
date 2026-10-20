Loads an image into a frame of the window -- harwood's own, so there is no
ds9 to start -- with the display limits chosen as IRAF's `display` chooses
them (`--scale zscale`, the default). With `--coords`, each position of a
list is circled, which is what `tvmark` is mostly used for: seeing what
daofind found and what it missed.

```console
$ harwood display red/obj001.fits --coords obj001.coo
$ harwood display red/obj002.fits --frame 2
```

Frames are numbered as in IRAF; the toolbar's *frame* picks one, *blink*
alternates between them and *tile* shows them side by side. From the
window's prompt, `display` loads into the window you are looking at; from a
terminal, it opens one, and `--no-wait` returns at once instead of waiting
for it to be closed.

Without a screen -- over ssh, in a script -- the picture is written to a
PNG instead, and the command says where. [impicture](task:impicture) writes
one on purpose.
