"""reidentify against IRAF's own reidentify.

tests/data/iraf_reidentify holds what IRAF 2.18.1's reidentify wrote for
tests/iraf_field.multispec_night() under nine sets of parameters: one
aperture of a multispec arc carried to the others and to a second arc
(with a shift of zero, with the shift found from the spectrum, adding
features from the line list), and a long slit reidentified line by line --
traced and not, with the shift carried from line to line, with a shift per
step, without refitting. Every case starts from the record IRAF's
autoidentify wrote, as IRAF's did.

The log has to come out the same character for character, and every record
with the same lines, fitted to IRAF's printed precision, and the same ones
rejected.
"""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import multispec_night

from harwood.core import database, spectrum
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "iraf_reidentify" / "cases.json").read_text(encoding="utf-8"))


def write_night(directory: Path) -> None:
    """The arcs IRAF was run on: arc3 and arc3b as harwood spectra, arc2d
    as the image it is."""
    night, frame = multispec_night()
    for name in ("arc3", "arc3b"):
        apertures, header = night[name]
        spectrum.write(directory / f"{name}.fits",
                       [spectrum.Aperture(number=ap, centre=centre, low=-5.0, high=5.0,
                                          pixel=np.arange(1.0, flux.size + 1.0),
                                          flux=flux.astype(float))
                        for ap, centre, flux in apertures],
                       header=fits.Header(header))
    fits.PrimaryHDU(frame, fits.Header({"OBJECT": "long slit arc", "DISPAXIS": 1})) \
        .writeto(directory / "arc2d.fits")
    shutil.copy(DATA / "arc_lines.dat", directory)


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_reidentify_reidentifies_what_iraf_reidentifies(tmp_path, monkeypatch, case):
    entry = REFERENCE["cases"][case]
    iraf = DATA / "iraf_reidentify" / case / "database"
    write_night(tmp_path)
    (tmp_path / "database").mkdir()
    name = f"id{entry['reference']}"
    autoidentified = re.split(r"(?m)^(?=# )", (iraf / name).read_text(encoding="utf-8"))[1]
    (tmp_path / "database" / name).write_text(autoidentified, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = get_task("reidentify").run(reference=entry["reference"], images=entry["images"],
                                        coordlist="arc_lines.dat", logfiles="",
                                        **entry["params"])
    assert result.data["text"] == entry["log"]

    for image in entry["images"].split(","):
        ours = {record.section: record for record in database.read_all(image, ".")}
        theirs = {record.section: record for record in database.read_all(image, iraf.parent)}
        assert ours.keys() == theirs.keys()
        for section, t in theirs.items():
            m = ours[section]
            assert len(m) == len(t), f"{image}{section}: a different set of lines"
            assert (m.used == t.used).all(), f"{image}{section}: other lines rejected"
            assert np.abs(m.pixel - t.pixel).max() <= 0.0051         # IRAF prints %.2f
            assert np.abs(m.fitted - t.fitted).max() < 5e-5
