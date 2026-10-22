"""SCI, VAR and DQ: the image as harwood keeps it (design contract, section 12).

Checked by hand against IRAF 2.18 when this was written: IRAF opens
x.fits[SCI] and inherits the primary header (INHERIT = T), takes the title
from OBJECT in the extension itself -- hence the copy -- and imcombine with
masktype=goodvalue rejects the pixels of the FITS mask --format iraf names in
BPM.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import dq, images, mef


@pytest.fixture
def image():
    rng = np.random.default_rng(5)
    sci = (100.0 + rng.normal(0, 2, (12, 16))).astype(np.float32)
    var = np.full(sci.shape, 4.0)
    quality = dq.empty(sci.shape)
    quality[3, 4] = dq.SATURATED | dq.COSMIC_RAY
    quality[7, 9] = dq.BAD_PIXEL
    primary = fits.Header({"OBJECT": "M92", "EXPTIME": 60.0, "FILTER": "V"})
    plane_header = fits.Header({"CRPIX1": 8.0, "CRVAL1": 1.0, "CDELT1": 1.0})
    return mef.Image(primary, [mef.Plane(sci, var, quality, plane_header)])


@pytest.mark.parametrize("uncertainty", ["VAR", "ERR", "UNCERT"])
def test_every_way_of_writing_the_uncertainty_reads_back_as_a_variance(
        tmp_path, image, uncertainty):
    mef.write(tmp_path / "x.fits", image, uncertainty=uncertainty)
    back = mef.read(tmp_path / "x.fits").plane
    np.testing.assert_allclose(back.var, 4.0, rtol=1e-6)
    np.testing.assert_array_equal(back.sci, image.plane.sci)
    np.testing.assert_array_equal(back.dq, image.plane.dq)       # every reason, not only bad


def test_the_extensions_are_named_as_section_12_says(tmp_path, image):
    mef.write(tmp_path / "x.fits", image)
    with fits.open(tmp_path / "x.fits") as hdul:
        assert hdul[0].data is None
        assert [(h.name, h.header["EXTVER"]) for h in hdul[1:]] == \
            [("SCI", 1), ("VAR", 1), ("DQ", 1)]
        assert hdul["DQ"].header["DQCONV"] == "DRAGONS"
        assert (hdul["DQ"].data.dtype.kind, hdul["DQ"].data.dtype.itemsize) == ("u", 2)
    mef.write(tmp_path / "e.fits", image, uncertainty="ERR")
    with fits.open(tmp_path / "e.fits") as hdul:
        np.testing.assert_allclose(hdul["ERR"].data, 2.0)


def test_astropy_reads_the_uncert_layout_as_its_own(tmp_path, image):
    from astropy.nddata import CCDData, StdDevUncertainty

    mef.write(tmp_path / "c.fits", image, uncertainty="UNCERT")
    ccd = CCDData.read(tmp_path / "c.fits", unit="adu")
    assert isinstance(ccd.uncertainty, StdDevUncertainty)
    np.testing.assert_allclose(ccd.uncertainty.array, 2.0)
    assert ccd.mask[3, 4] and ccd.mask[7, 9] and ccd.mask.sum() == 2


def test_a_task_reading_sci_still_sees_the_exposure(tmp_path, image):
    """images.read is what every task calls; with INHERIT it gets OBJECT,
    EXPTIME and FILTER from the primary and its own cards from SCI."""
    mef.write(tmp_path / "x.fits", image)
    data, header = images.read(tmp_path / "x.fits")
    np.testing.assert_array_equal(data, image.plane.sci)
    assert header["EXPTIME"] == 60.0 and header["FILTER"] == "V"
    assert header["CRPIX1"] == 8.0
    assert "EXTNAME" not in header, "a task writing it out must not make an extension"


def test_the_plane_wins_where_both_headers_have_a_card():
    primary = fits.Header({"OBJECT": "night", "BUNIT": "adu"})
    plane = fits.Header({"BUNIT": "electrons"})
    plane.add_history("from SCI")
    primary.add_history("from the primary")
    out = mef.merged(primary, plane)
    assert out["BUNIT"] == "electrons"
    assert list(out["HISTORY"]) == ["from the primary", "from SCI"]


def test_an_iraf_frame_is_read_as_it_is_with_its_mask(tmp_path):
    """One HDU, no variance (nobody can make one without the gain), and the
    DQ from the mask its BPM names."""
    sci = np.full((6, 8), 50.0, dtype=np.float32)
    mask = np.zeros((6, 8), dtype=np.int16)
    mask[2, 3] = 1
    fits.PrimaryHDU(mask).writeto(tmp_path / "bad.fits")
    fits.PrimaryHDU(sci, fits.Header({"BPM": "bad.fits", "OBJECT": "x"})) \
        .writeto(tmp_path / "raw.fits")
    read = mef.read(tmp_path / "raw.fits", workdir=tmp_path)
    assert read.layout == "single"
    assert read.plane.var is None
    assert read.plane.dq[2, 3] == dq.BAD_PIXEL and read.plane.dq.sum() == dq.BAD_PIXEL
    assert read.primary["OBJECT"] == "x"


def test_format_iraf_writes_one_hdu_and_the_dq_as_its_bpm(tmp_path, image):
    written, lost = mef.write(tmp_path / "i.fits", image, format="iraf")
    assert [p.name for p in written] == ["i.fits", "i.bpm.fits"]
    assert lost == ["the variance"], "what IRAF cannot hold is said, not dropped"
    with fits.open(tmp_path / "i.fits") as hdul:
        assert len(hdul) == 1
        assert hdul[0].header["BPM"] == "i.bpm.fits"
        assert hdul[0].header["FILTER"] == "V" and hdul[0].header["CRPIX1"] == 8.0
    np.testing.assert_array_equal(fits.getdata(tmp_path / "i.bpm.fits"), image.plane.dq)
    # And back: the mask becomes DQ bits again, as 'bad pixel'.
    back = mef.read(tmp_path / "i.fits", workdir=tmp_path).plane
    assert back.dq[3, 4] & dq.BAD_PIXEL and back.dq[7, 9] & dq.BAD_PIXEL


def test_planes_repeat_with_extver(tmp_path, image):
    two = mef.Image(image.primary, [image.plane, mef.Plane(image.plane.sci * 2)])
    mef.write(tmp_path / "m.fits", two)
    back = mef.read(tmp_path / "m.fits")
    assert len(back.planes) == 2
    assert back.planes[1].var is None, "no variance written for a plane without one"
    with pytest.raises(ValueError, match="planes"):
        _ = back.plane
    with pytest.raises(ValueError, match="one image per file"):
        mef.write(tmp_path / "m2.fits", back, format="iraf")


def test_the_dq_says_why_in_words():
    assert dq.describe(0) == "good"
    assert dq.describe(dq.SATURATED | dq.COSMIC_RAY) == "saturated, cosmic ray"
    assert dq.describe(256 | dq.BAD_PIXEL) == "bad pixel, other (256)"


def test_an_unknown_choice_is_refused_with_the_ones_there_are(tmp_path, image):
    with pytest.raises(ValueError) as raised:
        mef.write(tmp_path / "x.fits", image, uncertainty="SIGMA")
    assert "VAR" in raised.value.fix and "ERR" in raised.value.fix
