"""imcombine and its ccdred presets, against IRAF's own output.

tests/iraf_field.combine_stack() builds seven frames -- different sky
levels and exposure times, a star, cosmic rays in single frames, a few low
pixels. tests/data/iraf_imcombine.npz holds what IRAF 2.18.1 wrote for them:
images.imcombine over every combine and rejection algorithm and the scaling,
weighting, threshold, grow and nkeep options, with its sigma images; and
ccdred's zerocombine, darkcombine, flatcombine and combine, which run a
separate copy of the code with scaling of their own.

The tolerance is IRAF's float32 output: a quarter of a thousandth of a count
on levels of a few hundred.

One known difference, kept visible: ccdred's mode. On these frames the
installed IRAF 2.18.1 binary's flatcombine finds modes up to 0.2% away from
what its own published source computes (iraf-community, noao/imred/ccdred/
src/combine/icstat.gx, unchanged since 2017) -- the same source that, with
imcombine's constants, reproduces images.imcombine's modes exactly. harwood
follows the source. Everything else in flatcombine -- the avsigclip rejection
with its slips, the scaling by exposure -- matches the binary.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import combine_stack

from harwood.core.registry import get_task

REFERENCE = np.load(Path(__file__).parent / "data" / "iraf_imcombine.npz")

IMCOMBINE = {
    "avg_none": dict(combine="average", reject="none"),
    "med_none": dict(combine="median", reject="none"),
    "lmed_none": dict(combine="lmedian", reject="none"),
    "sum_none": dict(combine="sum", reject="none"),
    "avg_minmax": dict(combine="average", reject="minmax", nlow=1, nhigh=2),
    "avg_sigclip": dict(combine="average", reject="sigclip"),
    "avg_sigclip_nomclip": dict(combine="average", reject="sigclip", mclip=False),
    "avg_avsigclip": dict(combine="average", reject="avsigclip"),
    "avg_ccdclip": dict(combine="average", reject="ccdclip", rdnoise=8.0, gain=1.0),
    "avg_crreject": dict(combine="average", reject="crreject", rdnoise=8.0, gain=1.0),
    "avg_pclip": dict(combine="average", reject="pclip"),
    "med_sigclip": dict(combine="median", reject="sigclip"),
    "avg_mode_scale": dict(combine="average", reject="none", scale="mode"),
    "avg_median_zero": dict(combine="average", reject="none", zero="median"),
    "avg_mean_scale_sig": dict(combine="average", reject="sigclip", scale="mean"),
    "avg_exp_scale": dict(combine="average", reject="minmax", scale="exposure",
                          exposure_key="EXPTIME"),
    "avg_mode_scale_avsig": dict(combine="average", reject="avsigclip", scale="mode"),
    "avg_weight": dict(combine="average", reject="none", scale="mode", weight="mode"),
    "avg_thresh": dict(combine="average", reject="none", lthreshold=0, hthreshold=2500),
    "avg_sig_grow": dict(combine="average", reject="sigclip", grow=1),
    "avg_nkeep": dict(combine="average", reject="sigclip", lsigma=1, hsigma=1, nkeep=5),
    # grow where the rejection rejects: avg_sig_grow above rejects nothing on
    # this stack (seven frames cannot clip one outlier at 3 sigma when the
    # outlier is in the sigma), which is how a broken second pass went unseen.
    "avg_cr_grow1": dict(combine="average", reject="crreject", rdnoise=8.0, gain=1.0,
                         grow=1),
    "avg_cr_grow2": dict(combine="average", reject="crreject", rdnoise=8.0, gain=1.0,
                         grow=2.5),
    "med_ccd_grow": dict(combine="median", reject="ccdclip", rdnoise=8.0, gain=1.0,
                         grow=1.5, lsigma=2, hsigma=2),
    "avg_pclip_grow": dict(combine="average", reject="pclip", grow=1, lsigma=2, hsigma=2),
}

#: ccdred's combine, called through the flatcombine preset with its scaling
#: switched off or set, to check the rejection apart from the mode.
CCDRED = {
    "cc_avs": dict(reject="avsigclip", scale="none"),
    "cc_avs_nk": dict(reject="avsigclip", scale="none", lsigma=1.0, hsigma=1.0, nkeep=5),
    "cc_sig": dict(reject="sigclip", scale="none", lsigma=1.5, hsigma=1.5),
    "cc_ccd": dict(reject="ccdclip", scale="none", rdnoise=8.0, lsigma=1.5, hsigma=1.5),
    "cc_avs_exp": dict(reject="avsigclip", scale="exposure"),
}


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    where = tmp_path_factory.mktemp("stack")
    paths = []
    for index, (data, exptime) in enumerate(combine_stack()):
        path = where / f"f{index}.fits"
        fits.writeto(path, data, fits.Header({"EXPTIME": exptime, "OBJECT": "stack"}))
        paths.append(str(path))
    return where, paths


@pytest.mark.parametrize("case", list(IMCOMBINE))
def test_imcombine_writes_what_iraf_writes(frames, case):
    where, paths = frames
    params = dict(nlow=1, nhigh=1) | IMCOMBINE[case]
    get_task("imcombine").run(input=paths, output=where / f"{case}.fits",
                              sigmas=where / f"{case}_sigma.fits", **params)
    ours = fits.getdata(where / f"{case}.fits").astype(float)
    sigma = fits.getdata(where / f"{case}_sigma.fits").astype(float)
    assert np.abs(ours - REFERENCE[case]).max() < 2e-3
    assert np.abs(sigma - REFERENCE[case + "__sigma"]).max() < 4e-3


@pytest.mark.parametrize("task", ["zerocombine", "darkcombine"])
def test_the_ccdred_presets_are_iraf_s(frames, task):
    """IRAF's own values -- nlow 0, nhigh 1 -- and ccdred's scaling, which
    refers to the mean exposure: a combined dark comes out at 12.57 s, not at
    the first frame's 10."""
    where, paths = frames
    out = where / f"P_{task}.fits"
    get_task(task).run(input=paths, output=out, ccdtype="", subsets=False)
    assert np.abs(fits.getdata(out) - REFERENCE[f"P_{task}"]).max() < 2e-3
    header = fits.getheader(out)
    assert header["EXPTIME"] == pytest.approx(12.57143, abs=1e-4)
    assert header["NCOMBINE"] == 7


@pytest.mark.parametrize("case", list(CCDRED))
def test_ccdred_rejection_is_iraf_s(frames, case):
    """ccdred's copy of the rejection, slips included: avsigclip's line sigma
    reused by the nkeep add-back, and the median loops bounded by a count."""
    from harwood.tasks.imcombine import FlatCombine

    where, paths = frames
    out = where / f"{case}.fits"
    FlatCombine().run(input=paths, output=out, ccdtype="", subsets=False, blank=0.0,
                      **CCDRED[case])
    assert np.abs(fits.getdata(out) - REFERENCE[case]).max() < 2e-3


def test_flatcombine_with_its_mode_scaling_stays_close(frames):
    """The known difference: the installed binary's ccdred mode. The combined
    flat agrees to within 1%, and in its median to 0.2%."""
    where, paths = frames
    out = where / "P_flatcombine.fits"
    get_task("flatcombine").run(input=paths, output=out, ccdtype="", subsets=False)
    ours = fits.getdata(out).astype(float)
    theirs = REFERENCE["P_flatcombine"].astype(float)
    assert np.max(np.abs(ours / theirs - 1)) < 0.01
    assert abs(np.median(ours / theirs) - 1) < 0.002


def test_the_presets_have_no_grow_or_weight_as_in_iraf(frames):
    where, paths = frames
    with pytest.raises(ValueError) as raised:
        get_task("zerocombine").run(input=paths, output=where / "z.fits", ccdtype="",
                                    grow=1)
    assert "imcombine" in raised.value.fix


def test_grow_is_exercised_by_the_grow_cases(frames):
    """Each grow case must differ from the same rejection without grow, or it
    tests nothing."""
    where, paths = frames
    for case in ("avg_cr_grow1", "avg_cr_grow2", "med_ccd_grow", "avg_pclip_grow"):
        params = dict(nlow=1, nhigh=1) | IMCOMBINE[case] | {"grow": 0.0}
        get_task("imcombine").run(input=paths, output=where / f"{case}_nogrow.fits",
                                  **params)
        without = fits.getdata(where / f"{case}_nogrow.fits")
        assert np.abs(without - REFERENCE[case]).max() > 1.0, case
