"""imrename: IRAF's rules for the new name, and never a clobbered image.

What IRAF 2.18.1 did, renaming a.fits to q.txt, b.fits to r.imh, f.fit to g
and c.fits onto an existing e.fits: q.txt.fits, r.fits, g.fit, and a refusal.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core.registry import get_task


@pytest.fixture
def night(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("a.fits", "b.fits", "c.fits", "e.fits", "f.fit"):
        fits.writeto(name, np.zeros((2, 2), np.float32))
    return tmp_path


def test_the_new_name_follows_iraf(night):
    get_task("imrename").run(oldnames=["a.fits", "b.fits", "f.fit"],
                             newnames=["q.txt", "r.imh", "g"])
    assert {p.name for p in night.iterdir()} >= {"q.txt.fits", "r.fits", "g.fit"}
    assert not (night / "a.fits").exists()


def test_an_existing_image_is_never_overwritten_and_nothing_moves(night):
    with pytest.raises(ValueError, match="overwrite e.fits"):
        get_task("imrename").run(oldnames=["a.fits", "c.fits"], newnames=["z", "e"])
    assert (night / "a.fits").exists() and not (night / "z.fits").exists()


def test_renaming_an_image_onto_itself_is_allowed(night):
    get_task("imrename").run(oldnames=["a.fits"], newnames=["a"])
    assert (night / "a.fits").exists()


def test_a_directory_takes_the_images_with_their_names(night):
    (night / "raw").mkdir()
    get_task("imrename").run(oldnames=["a.fits", "b.fits"], newnames=["raw"])
    assert sorted(p.name for p in (night / "raw").iterdir()) == ["a.fits", "b.fits"]


def test_the_lists_must_pair_up_and_whole_files_move(night):
    with pytest.raises(ValueError, match="different number"):
        get_task("imrename").run(oldnames=["a.fits", "b.fits"], newnames=["x"])
    with pytest.raises(ValueError, match="part of a file"):
        get_task("imrename").run(oldnames=["a.fits[1:1,*]"], newnames=["x"])
