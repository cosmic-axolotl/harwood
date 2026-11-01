"""The sessions of ccdinstrument's gabarito: what is typed, what is left behind."""

ALL = "a.fits,b.fits,c.fits"
CASES = {
    "table-basic": dict(images=ALL, edit=False),
    "table-common": dict(images=ALL, edit=False, parameters="common"),
    "table-all": dict(images=ALL, edit=False, parameters="all"),
    "no-instrument": dict(images="a.fits,d.fits", edit=False, instrument="", parameters="all"),
    "no-file": dict(images="a.fits", edit=False, instrument="nothere.dat"),
    "missing-image": dict(images="nothing.fits,a.fits", edit=False),
    "scaled": dict(images="d.fits,e.fits,m.fits[1]", edit=False, parameters="all"),
    "help-1": dict(images="a.fits", stdin="?\nquit\n"),
    "help-2": dict(images="a.fits", stdin="help\nquit\n", parameters="common"),
    "help-3": dict(images="a.fits", stdin="?\nq\n", parameters="all"),
    "header": dict(images="a.fits", stdin="imheader\nnewimage d.fits\nimh\nq\n"),
    "header-mef": dict(images="m.fits[1]", stdin="imh\nq\n"),
    "header-scaled": dict(images="e.fits,b.fits,c.fits", stdin="imh\nn\nimh\nn\nimh\nq\n"),
    "instrument": dict(images="a.fits", stdin="instrument\ni\nq\n"),
    "ask-keyword": dict(
        images="a.fits", stdin="exptime\nEXPTIME\nexptime\n\nexptime TEL\nq\nyes\n"
    ),
    "keyword-default": dict(
        images="a.fits", stdin="darktime dark 3\nexptime x 'a b'\nq\nno\n"
    ),
    "ambiguous": dict(images="a.fits", stdin="bogus\nt\nd\ndar\n\ndark\ndark\n\n\nq\n"),
    "translate": dict(images="a.fits", stdin="translate flat\nq\nno\n"),
    "translate-ask": dict(
        images="b.fits", stdin="translate\nzero\ntr\n\ntr nonsense\nq\nno\n"
    ),
    "translate-none": dict(images="d.fits", stdin="translate\ndark\nq\nno\n"),
    "write": dict(images="a.fits", stdin="subset FILTER 'a b'\nwrite o.dat\nq\n"),
    "write-ask": dict(images="a.fits", stdin="exptime A\nwrite\nout.dat\nq\n"),
    "read-other": dict(images="a.fits", stdin="exptime A\nread o2.dat\ninstrument\nq\nno\n"),
    "read-missing": dict(images="a.fits", stdin="read nofile.dat\ninstrument\nq\nno\n"),
    "newimage": dict(images="a.fits", stdin="newimage nothing.fits\nn\nb.fits\nn\n\nq\n"),
    "next": dict(images="a.fits,b.fits,c.fits", stdin="next\nnext\nexptime Q\nq\nno\n"),
    "quit-early": dict(images="a.fits,b.fits", stdin="q\n"),
    "eof": dict(images="a.fits,b.fits", stdin=""),
    "eof-prompt": dict(images="a.fits", stdin="exptime"),
    "update-yes": dict(images="a.fits", stdin="exptime B\nq\nyes\n"),
    "update-blank": dict(images="a.fits", stdin="exptime B\nq\n\n"),
    "update-eof": dict(images="a.fits", stdin="exptime B\nq\n"),
    "update-other": dict(images="a.fits", stdin="exptime B\nq\nmaybe\n"),
    "show": dict(images="a.fits", stdin="show\nimagetyp\n\nsubset\n\nq\n", parameters="all"),
    "flags": dict(
        images="a.fits",
        stdin="flatcor\n\nflatcor FLATCOR done\nfixpix TOOBIG a\ntrimsec CCDSEC\n"
        "ccdmean EXPTIME\ndate-obs\n\nra\n\ntitle\n\nq\nno\n",
    ),
    "subsets": dict(images="a.fits,b.fits,c.fits,a.fits", stdin="next\nnext\nnext\nq\n"),
}
