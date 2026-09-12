"""What section 7.5 requires of the compiled layer.

The demanding test is not that the means agree: it is that the **rejection
counts** agree. Clipping makes discrete decisions, and a backend can reject a
different pixel while landing inside any tolerance you would put on the mean.
That is precisely the bug that would never be noticed.
"""

import numpy as np
import pytest

from harwood import kernels
from harwood.kernels import reference

BACKENDS = sorted(kernels.available())


@pytest.fixture
def stack():
    rng = np.random.default_rng(20260912)
    data = rng.normal(1000.0, 10.0, (15, 64, 64))
    data[4, 20, 20] = 6000.0          # one cosmic ray
    data[9, 30, 30] = 5800.0          # another, elsewhere
    data[2, 40, 40] = -400.0          # a low outlier too
    return data


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_agree_on_every_rejection(backend, stack):
    want_mean, want_nrej = reference.sigclip_mean(stack, 3.0, 3.0, maxiter=5)
    kernel = kernels.get("sigclip_mean", backend=backend)
    got_mean, got_nrej = kernel(stack, 3.0, 3.0, maxiter=5)

    np.testing.assert_array_equal(got_nrej, want_nrej)
    np.testing.assert_allclose(got_mean, want_mean, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_agree_on_the_noise_model(backend, stack):
    want_mean, want_nrej = reference.ccdclip_mean(stack, gain=2.0, rdnoise=6.5)
    kernel = kernels.get("ccdclip_mean", backend=backend)
    got_mean, got_nrej = kernel(stack, gain=2.0, rdnoise=6.5)

    np.testing.assert_array_equal(got_nrej, want_nrej)
    np.testing.assert_allclose(got_mean, want_mean, rtol=1e-12, atol=1e-9)


def test_sigclip_finds_a_lone_cosmic_ray(stack):
    mean, nrej = reference.sigclip_mean(stack, 3.0, 3.0)
    assert nrej[20, 20] == 1
    assert mean[20, 20] == pytest.approx(1000.0, abs=5.0)


def test_two_cosmic_rays_on_one_pixel_defeat_sigclip():
    """The known limitation, kept as a test rather than as a footnote.

    Section 7.5: with few images, two deviant values on the same pixel inflate
    the stack's own spread until the rejection window swallows them both. This
    is *why* IRAF offers ccdclip, and if this test ever starts passing the
    other way, the documentation is wrong.
    """
    rng = np.random.default_rng(7)
    stack = rng.normal(1000.0, 10.0, (15, 8, 8))
    stack[3, 4, 4] = 6000.0
    stack[7, 4, 4] = 5800.0

    mean, nrej = reference.sigclip_mean(stack, 3.0, 3.0)
    assert nrej[4, 4] == 0                       # nothing rejected
    assert mean[4, 4] > 1500.0                   # and the mean is wrong

    mean, nrej = reference.ccdclip_mean(stack, gain=2.0, rdnoise=6.5)
    assert nrej[4, 4] == 2                       # the noise model catches both
    assert mean[4, 4] == pytest.approx(1000.0, abs=5.0)


def test_median_centring_survives_an_extreme_outlier():
    """Without mclip, a strong deviant drags the mean and everything is rejected."""
    rng = np.random.default_rng(3)
    stack = rng.normal(500.0, 5.0, (9, 4, 4))
    stack[0, 2, 2] = 1e6

    _, nrej = reference.ccdclip_mean(stack, gain=2.0, rdnoise=6.0, mclip=True)
    assert nrej[2, 2] == 1

    _, nrej_nomedian = reference.ccdclip_mean(stack, gain=2.0, rdnoise=6.0, mclip=False)
    assert nrej_nomedian[2, 2] > 1               # collateral damage, as expected


def test_a_pixel_with_two_values_is_never_emptied():
    stack = np.array([[[10.0]], [[1000.0]]])
    mean, nrej = reference.sigclip_mean(stack, 3.0, 3.0)
    assert nrej[0, 0] == 0
    assert mean[0, 0] == pytest.approx(505.0)


def test_bad_pixel_mask_is_honoured():
    stack = np.ones((5, 3, 3)) * 100.0
    stack[0, 1, 1] = 9999.0
    mask = np.zeros(stack.shape, dtype=bool)
    mask[0, 1, 1] = True

    mean, nrej = reference.sigclip_mean(stack, 3.0, 3.0, mask=mask)
    assert mean[1, 1] == pytest.approx(100.0)
    assert nrej[1, 1] == 1


def test_unknown_backend_is_refused_loudly(monkeypatch):
    monkeypatch.setenv("HARWOOD_BACKEND", "julia")
    with pytest.raises(RuntimeError, match="unknown backend"):
        kernels.get("sigclip_mean")


def test_requested_backend_is_never_silently_downgraded(monkeypatch):
    monkeypatch.setenv("HARWOOD_BACKEND", "rust")
    if "rust" in kernels.available():
        pytest.skip("rust kernels are installed here")
    with pytest.raises(RuntimeError, match="not installed"):
        kernels.get("sigclip_mean")


def test_kernel_reports_its_backend_for_the_log():
    kernel = kernels.get("sigclip_mean", backend="reference")
    assert kernel.backend == "reference"
    assert kernel.label.startswith("reference ")
