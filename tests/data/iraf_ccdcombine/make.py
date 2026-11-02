"""The frames of tests/data/iraf_ccdcombine/cases.py."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parents[2]))
from iraf_field import combine_stack  # noqa: E402


def write(where):
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    names = []
    for k, (data, exptime) in enumerate(combine_stack()):
        fits.PrimaryHDU(np.asarray(data, np.float32), fits.Header(
            {"EXPTIME": exptime, "OBJECT": "stack"})).writeto(where / f"f{k}.fits",
                                                              overwrite=True)
        names.append(f"f{k}.fits")
    (where / "all.lis").write_text("\n".join(names) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write(sys.argv[1])
