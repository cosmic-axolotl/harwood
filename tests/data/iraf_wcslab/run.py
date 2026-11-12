"""IRAF 2.18's wcslab, for tests/test_wcslab_iraf.py:

    python wcl_inputs.py <folder>                                      (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (PyRAF)
    python wcl_collect.py <results>                                    (harwood's)

Each case in a process of its own, wcslab, wlpars and wcspars unlearned, the plot
to stdvdm (kept as vdm), both streams kept."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from wcl_cases import CASES

WLPARS = [
    "major_grid", "minor_grid", "dolabel", "remember", "axis1_beg", "axis1_end",
    "axis1_int", "axis2_beg", "axis2_end", "axis2_int", "major_line", "major_tick",
    "axis1_minor", "axis2_minor", "minor_line", "minor_tick", "tick_in", "axis1_side",
    "axis2_side", "axis2_dir", "justify", "labout", "rotate", "full_label",
    "label_size", "title", "axis1_title", "axis2_title", "title_side",
    "axis1_title_side", "axis2_title_side", "title_size", "axis_title_size",
    "graph_type"]
WCSPARS = [
    "ctype1", "ctype2", "crpix1", "crpix2", "crval1", "crval2", "cd1_1", "cd1_2",
    "cd2_1", "cd2_2", "log_x1", "log_x2", "log_y1", "log_y2"]


def one(case: str) -> None:
    from pyraf import iraf

    image, params = CASES[case]
    iraf.images(_doprint=0)
    iraf.tv(_doprint=0)
    for task in ("wcslab", "wlpars", "wcspars"):
        iraf.unlearn(task)
    own = {}
    for key, value in params.items():
        if key in WLPARS:
            iraf.wlpars.setParam(key, value)
        elif key in WCSPARS:
            iraf.wcspars.setParam(key, value)
        else:
            own[key] = value
    vdm = Path(iraf.osfn("uparm$")) / "vdm"
    vdm.unlink(missing_ok=True)
    try:
        iraf.wcslab(image, 1, device="stdvdm", Stdout="out.txt", Stderr="err.txt", **own)
    except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
        with open("err.txt", "a", encoding="utf-8") as fd:
            fd.write(f"ERROR: {error}\n")
    if vdm.exists():
        shutil.copy(vdm, "vdm")
    iraf.flpr()


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2])
        sys.exit()
    here = Path(__file__).resolve().parent
    fresh, results = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    only = sys.argv[3:] or list(CASES)
    for case in only:
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        done = subprocess.run([sys.executable, str(here / "run.py"), "--one", case],
                              cwd=work, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=300,
                              env={**os.environ, "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        shutil.rmtree(work / "pyraf", ignore_errors=True)
        for fits in work.glob("*.fits"):
            fits.unlink()
        print(case, done.returncode, flush=True)
