"""The cases of IRAF's scoords gabarito, on spectra of 30 pixels."""

import numpy as np

N = 30
WAVES = 4000.0 + 3.1 * np.arange(N) + 0.0123456789 * np.arange(N) ** 2
FILES = {
    "good.dat": "\n".join(repr(float(w)) for w in WAVES) + "\n",
    "words.dat": "# wavelengths\n" + "\n".join(f"{w:.6f} extra" for w in WAVES)
                 + "\nnot a number\n",
    "short.dat": "\n".join(f"{w}" for w in WAVES[:-2]) + "\n",
    "long.dat": "\n".join(f"{w}" for w in np.r_[WAVES, 9000.0, 9001.0]) + "\n",
    "back.dat": "\n".join(f"{w}" for w in np.r_[WAVES[:10], WAVES[8], WAVES[11:]]) + "\n",
    "firststep.dat": "\n".join(f"{w}" for w in np.r_[WAVES[1], WAVES[0], WAVES[2:]]) + "\n",
}
CASES = {
    "good": ("s1", "good.dat"),
    "words": ("s2", "words.dat"),
    "short": ("s3", "short.dat"),
    "long": ("s4", "long.dat"),
    "back": ("s5", "back.dat"),
    "firststep": ("s6", "firststep.dat"),
}
