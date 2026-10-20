"""asttimes against IRAF's own output (tests/data/iraf_asttimes.json): zone
time to UT, epoch, Julian date and local mean sidereal time, with the day of
the week, across a change of day."""

import json
from pathlib import Path

import pytest

from harwood.core.registry import get_task

DATA = json.loads((Path(__file__).parent / "data" / "iraf_asttimes.json").read_text())


@pytest.mark.parametrize("site", ["kpno", "lna"])
def test_asttimes_prints_irafs_table(site, tmp_path):
    dates = tmp_path / "dates.txt"
    dates.write_text("\n".join(row[0] for row in DATA[site]) + "\n")
    rows = get_task("asttimes").run(files=[str(dates)], observatory=site).data["rows"]
    for row, expected in zip(rows, DATA[site], strict=True):
        assert list(row.values()) == expected[1:]


def test_a_julian_date_and_a_sidereal_time_turn_back_into_times():
    back = get_task("asttimes").run(jd=2448646.5833333, observatory="kpno").data["rows"][0]
    assert (back["date"], back["ut"]) == ("1992-01-24", "02:00:00.0")
    when = get_task("asttimes").run(date="1992-01-25", lmst="2:48:10.5",
                                    observatory="kpno").data["rows"][0]
    assert when["ut"] == "02:00:00.0" and when["lmst"] == "02:48:10.5"
