"""The spectra of the emplot gabarito: python ep_inputs.py <folder> (harwood's venv).

em_inputs.py's, NGC 660's nucleus once more with its dispersion running from red to
blue (CDELT1 < 0), which emplot handles apart, and once more with what xcsao writes
to the header (save_vel): the correlation's velocity and three templates', which
emsao reads (xcrhead) and its graph shows and switches between (t)."""

import sys
from pathlib import Path

import numpy as np
from em_inputs import NGC, linear, spectrum, write


def write_all(where: Path) -> None:
    write(where)
    rng = np.random.default_rng(31)
    w = 6958.9 - 1.1 * np.arange(600)
    linear("reversed", 6958.9, -1.1, spectrum(w, NGC, 0.0026, 2.2, 400.0, rng, 8.0),
           where, APID1="reversed")
    w = 6300.0 + 1.1 * np.arange(600)
    linear("xcor", 6300.0, 1.1, spectrum(w, NGC, 0.0026, 2.2, 400.0, rng, 8.0), where,
           APID1="xcor", CZXC=812.437, CZXCERR=21.25, CZXCR=7.31, BESTTEMP="femtemp97",
           CZPARAMS="3 2048 1 20 50 1000 2000 0.05",
           XCTEMP1="femtemp97 812.437 21.250 7.31 0.512 310.200 0.000 0.000 no 0",
           XCTEMP2="habtemp0 798.120 35.500 4.12 0.300 402.000 0.000 0.000 yes 1",
           XCTEMP3="eltemp 805.900 18.000 9.85 0.650 280.000 0.000 0.000 no 3")
    linear("xcor_bad", 6300.0, 1.1, spectrum(w, NGC, 0.0026, 2.2, 400.0, rng, 8.0),
           where, CZXC=1830.0, CZXCERR=15.0, CZXCR=12.0, BESTTEMP="sptemp",
           CZPARAMS="1 2048 1 20 50 1000 2000 0.05",
           XCTEMP1="sptemp 1830.000 15.000 12.00 0.700 300.000 0.000 0.000 no 2")


if __name__ == "__main__":
    write_all(Path(sys.argv[1]))
