"""photcal: the transformation from instrumental to standard magnitudes.

IRAF's noao.digiphot.photcal, from iraf-community 2.18, in the parts that do
the arithmetic: the configuration language (config), the catalog and
observation files (io), the fit of the transformation equations (fit, which
is fitparams over inlfit over nlfit) and their inversion (invert, which is
invertfit). All of it in single precision, as IRAF computes it, because
which iteration a fit stops at, and so the last digit of a coefficient,
depends on the rounding.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""
