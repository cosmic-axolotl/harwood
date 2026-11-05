"""The cases of IRAF's sinterp gabarito: a table of a smooth curve with a kink, the x's
asked for (inside, at its points, beyond its ends), each interpolant along a range,
and a spectrum."""

import numpy as np

TABLE = "\n".join(f"{x:.4f} {y:.5f}" for x, y in zip(
    np.linspace(3000, 9000, 25), 1.0 + 0.4 * np.sin(np.linspace(0, 3.1, 25))
    + np.where(np.linspace(3000, 9000, 25) > 6000, 0.15, 0.0), strict=True)) + \
    "\n# a comment\n17\n"
POINTS = "2500\n3000\n3125.5\n4250\n6000\n6100\n8999.9\n9600\n"
CASES = {
    "points_linear": dict(interp_mode="linear"),
    "points_curves": dict(interp_mode="curves"),
    "points_cheb": dict(interp_mode="chebyshev", order=5),
    "range_linear": dict(interp_mode="linear", curve_gen=True, x1=3100.0, x2=8900.0,
                         npts=30),
    "range_curves": dict(interp_mode="curves", curve_gen=True, x1=3100.0, dx=250.0, npts=24),
    "range_legendre": dict(interp_mode="legen", curve_gen=True, x1=3000.0, x2=9000.0,
                           dx=300.0, order=7),
    "range_spline3": dict(interp_mode="spline3", curve_gen=True, x1=3000.0, x2=9000.0,
                          npts=41, order=4),
    "range_spline1": dict(interp_mode="spline1", curve_gen=True, x1=3050.0, x2=8950.0,
                          npts=12, order=6),
    "image": dict(interp_mode="curves", make_image=True, x1=3000.0, x2=9000.0, npts=200),
}
