"""The ccdred package's parameters in ccdproc, IRAF's against harwood's: the output's
pixel type, the backup of an image processed in place, a bad pixel file of regions, and
the overscan fitted from a file of cursor keys with the answers to its questions.
name -> (images, ccdred's settings, ccdproc's parameters, cursor, answers).

pkg_inputs.write adds to ccdp_inputs' frames: sobj.fits (short), uobj.fits (ushort),
iobj.fits (int), dobj.fits (double), the same object in each; Bobj.fits, so that the
backup's first name is taken; bak/, a directory to back up into; badpix.dat."""

SEC = dict(overscan="yes", trim="yes", biassec="[53:60,1:40]", trimsec="[1:52,1:37]")

CASES = {}
for image in ("obj", "sobj", "uobj", "iobj", "dobj"):
    for kind in ("real", "short", "ushort", "int", "long", "double", ""):
        CASES[f"type-{image}-{kind or 'none'}"] = (
            f"{image}.fits", {"pixeltype": f"{kind} real".strip() if kind else ""}, SEC,
            "", "")
CASES["backup-prefix"] = ("obj.fits", {"backup": "B"}, SEC, "", "")
CASES["backup-free"] = ("comp.fits", {"backup": "B"}, SEC, "", "")
CASES["backup-dir"] = ("obj.fits", {"backup": "bak/"}, SEC, "", "")
CASES["backup-none"] = ("obj.fits", {"backup": ""}, SEC, "", "")
CASES["fix-text"] = ("obj.fits", {}, dict(fixpix="yes", fixfile="badpix.dat", **SEC), "", "")
CASES["fix-text-only"] = ("obj.fits", {}, dict(fixpix="yes", fixfile="badpix.dat"), "", "")
# the overscan by hand: three frames, the order raised in the first and kept for the
# next; no to the second; NO to the rest
CASES["hand"] = ("obj.fits,comp.fits,other.fits", {},
                 dict(interactive="yes", order=1, **SEC),
                 ":order 3\n0 0 1 f\n0 0 1 q\n0 0 1 q\n", "yes\nno\nYES\n")
CASES["hand-no"] = ("obj.fits,comp.fits,other.fits", {},
                    dict(interactive="yes", order=2, **SEC),
                    ":order 4\n0 0 1 f\n0 0 1 q\n", "yes\nNO\n")
CASES["hand-delete"] = ("obj.fits", {}, dict(interactive="yes", order=2, **SEC),
                        "20 0 1 d\n0 0 1 f\n0 0 1 q\n", "yes\n")
