"""identify moving among the lines of an image -- j, k, o -- against IRAF 2.18.

tests/data/iraf_identify_lines holds four sessions IRAF's identify ran: on a
multispec arc of three apertures (one at a time, whatever nsum says), with
nsum 10 there too, on a long slit nsum lines at a time and round its end, and
on the multispec arc again with the records of the first session already in
the database, and there the colon commands that read, add and write records
and set nsum. Each line left is kept, and comes back as it was; a line with a
record starts from it, one without from the features in hand; q writes this
line and each other one changed, the last kept first. What the keys print and
every record written must be IRAF's.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import multispec_night
from test_identify_keys import _same, _text

from harwood.core import database, icfit, identifysession, spectrum
from harwood.core import identify as idn
from harwood.core import reidentify as ri
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
KEYS = DATA / "iraf_identify_lines"
CASES = {"ms1": ("arc3", 1, None), "ms10": ("arc3", 10, None), "slit": ("arc2d", 10, None),
         "msdb": ("arc3", 1, "ms1"), "colon": ("arc3", 1, "ms1")}
#: IRAF writes REFSPEC1 into the arc's header with every record (id_dbwrite), and
#: says so when it cannot delete a REFSPEC2 it never had; harwood leaves the arc as
#: it is -- refspectra says which arc a spectrum takes.
REFSPEC = "Warning: Attempt to delete a nonexistent image parameter (REFSPEC2)\n"


def _records(path: Path) -> list[tuple[str, list[list[str]]]]:
    """IRAF's record file: each record's section and its feature rows."""
    out = []
    for block in re.split(r"(?m)^begin", path.read_text(encoding="utf-8"))[1:]:
        head = block.splitlines()[0].split("identify ", 1)[1].strip()
        rows = [line.split() for line in block.splitlines() if line.startswith("\t ")]
        rows = [row for row in rows if len(row) >= 6 and re.fullmatch(r"[-\d.]+", row[0])
                and row[4] in ("1", "2")]
        out.append((head, rows))
    return out


def _session(image, nsum, start, tmp_path):
    night, frame = multispec_night()
    if start:
        (tmp_path / "database").mkdir()
        shutil.copy(KEYS / f"{start}.db", tmp_path / "database" / f"id{image}")

    def read(name, section, _):
        found = [r for r in database.read_all(name, tmp_path) if r.section == section] \
            if (tmp_path / "database" / f"id{name}").exists() else []
        return ri.reference_from(found[-1], (section,)) if found else None

    if image == "arc3":
        apertures = night["arc3"][0]

        def load(line, _):
            number, _, flux = apertures[line[0] - 1]
            return flux, f" - Ap {number}"

        lines = identifysession.Lines((len(apertures), 1), (1, 1), (2, 1), load, read,
                                      kind="ms", numbers=[a[0] for a in apertures])
    else:
        count = frame.shape[0]

        def load(line, nsums):
            return idn.arc_vector(frame, f"line {line[0]}", nsums[0])

        lines = identifysession.Lines((count, 1), (min(count, nsum), 1),
                                      ((count + 1) // 2, 1), load, read)
    vector, section = lines.load(lines.line, lines.nsum)
    ident = idn.Identify(
        vector, idn.read_linelist(DATA / "arc_lines.dat"),
        icfit.FitSpec(function="spline3", order=1, sample="*", naverage=1, niterate=0,
                      low_reject=3.0, high_reject=3.0, grow=0.0),
        fwidth=4.0, ftype="emission", cradius=5.0, threshold=0.0, minsep=2.0, match=-3.0,
        maxfeatures=50)
    return identifysession.IdentifySession(
        ident, image, f"identify {image}{section}", "arc_lines.dat", "database",
        crval=4160.0, cdelt=0.29, section=section, lines=lines)


@pytest.mark.parametrize("case", list(CASES))
def test_the_lines_of_an_image_are_identified_as_iraf_does(case, tmp_path):
    image, nsum, start = CASES[case]
    session = _session(image, nsum, start, tmp_path)
    session.run_commands((KEYS / f"{case}.cur").read_text(encoding="utf-8"),
                         (KEYS / f"{case}.ans").read_text(encoding="utf-8"))
    ours = _text(session.stdout)
    theirs = _text((KEYS / f"{case}.out").read_text(encoding="utf-8"))
    assert len(ours) == len(theirs)
    for mine, iraf_s in zip(ours, theirs, strict=True):
        assert _same(mine, iraf_s), (mine, iraf_s)
    assert session.stderr == (KEYS / f"{case}.err").read_text(
        encoding="utf-8").replace(REFSPEC, "")

    # A database started from records keeps them; what IRAF added is after them,
    # what :write wrote first.
    written = _records(KEYS / f"{case}.db")
    if start:
        written = written[len(_records(KEYS / f"{start}.db")):]
    records = session.written + (session.records() if session.write else [])
    assert [r.section for r in records] == [head.replace(image, "", 1) for head, _ in written]
    for record, (_, rows) in zip(records, written, strict=True):
        assert len(rows) == len(record.features)
        for row, feature in zip(rows, record.features, strict=True):
            assert float(row[0]) == pytest.approx(feature.pix, abs=0.0051)
            assert float(row[1]) == pytest.approx(feature.fit, abs=5e-5)


def test_identify_writes_every_aperture_the_keys_changed(tmp_path, monkeypatch):
    night, _ = multispec_night()
    apertures, header = night["arc3"]
    spectrum.write(tmp_path / "arc3.fits",
                   [spectrum.Aperture(number=ap, centre=centre, low=-5.0, high=5.0,
                                      pixel=np.arange(1.0, flux.size + 1.0),
                                      flux=flux.astype(float))
                    for ap, centre, flux in apertures], header=fits.Header(header))
    shutil.copy(DATA / "arc_lines.dat", tmp_path)
    monkeypatch.chdir(tmp_path)
    get_task("identify").run(image="arc3.fits", coordlist="arc_lines.dat", crval=4160.0,
                             cdelt=0.29, nsum=1, gcommands=str(KEYS / "ms1.cur"),
                             answers=str(KEYS / "ms1.ans"))
    ours = {r.section: r for r in database.read_all("arc3", tmp_path)}
    assert sorted(ours) == [" - Ap 1", " - Ap 2", " - Ap 3"]
    for head, rows in _records(KEYS / "ms1.db"):
        record = ours[head.replace("arc3", "", 1)]
        assert len(record) == len(rows)
        assert np.abs(record.pixel - [float(r[0]) for r in rows]).max() <= 0.0051
        assert record.aperture == int(head.split()[-1])


def test_write_puts_the_record_in_the_database_at_once(tmp_path, monkeypatch):
    """:write writes now; with nothing changed after it, q has nothing to ask."""
    night, _ = multispec_night()
    apertures, header = night["arc3"]
    spectrum.write(tmp_path / "arc3.fits",
                   [spectrum.Aperture(number=ap, centre=centre, low=-5.0, high=5.0,
                                      pixel=np.arange(1.0, flux.size + 1.0),
                                      flux=flux.astype(float))
                    for ap, centre, flux in apertures], header=fits.Header(header))
    shutil.copy(DATA / "arc_lines.dat", tmp_path)
    (tmp_path / "database").mkdir()
    shutil.copy(KEYS / "ms1.db", tmp_path / "database" / "idarc3")
    monkeypatch.chdir(tmp_path)
    get_task("identify").run(image="arc3.fits", coordlist="arc_lines.dat", nsum=1,
                             gcommands=str(KEYS / "colon.cur"),
                             answers=str(KEYS / "colon.ans"))
    ours = {r.section: r for r in database.read_all("arc3", tmp_path)}
    assert len(ours[" - Ap 2"]) == 209, "Ap 1's lines read, Ap 3's added, written"
    assert len(ours[" - Ap 1"]) == 107 and len(ours[" - Ap 3"]) == 102
