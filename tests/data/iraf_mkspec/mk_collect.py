"""The pixels IRAF's mkspec wrote, to the bit, into mk.json beside run.py's titles.

    python mk_collect.py <scratch>                                  (harwood's)
"""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

work = Path(sys.argv[1]).resolve()
here = Path(__file__).with_name("mk.json")
out = json.loads(here.read_text(encoding="utf-8"))
for name in out:
    data = fits.getdata(work / f"{name}.fits")
    out[name]["shape"] = list(data.shape)
    out[name]["bits"] = np.asarray(data, np.float32).ravel().view(np.uint32).tolist()
here.write_text(json.dumps(out), encoding="utf-8")
