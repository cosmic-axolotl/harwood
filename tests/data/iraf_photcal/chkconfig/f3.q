

** Beginning of compilation **

** Error near line 12: set CV = v1 + VFIT
Set equation declared more than once [BV]
** Error near line 12: set CV = v1 + VFIT
Illegal identifier in set equation [v1]
** Error near line 12: set CV = v1 + VFIT
Undefined identifier in expression [VFIT]
** Error near line 13: VFIT : mV = V + v1 + v2 * BV + Q * XV
Undefined identifier in expression [Q]
** Error near line 15: BV : mB = B + b1
Transformation equation [VFIT] declared more than once
** Error near line 16: BFIT : mB + b1 = B + b1 + VFIT + BV
Transformation equation [BV] declared more than once
** Error near line 16: BFIT : mB + b1 = B + b1 + VFIT + BV
Illegal identifier in reference equation [b1]
** Error near line 16: BFIT : mB + b1 = B + b1 + VFIT + BV
Illegal identifier in fit equation [VFIT]
** Error near line 18: derivative (VFIT, w1) = 1
Derivative of undefined equation [WFIT]
** Error near line 19: derivative (BV, v1) = 1
Derivative with respect of undefined parameter [w1]
** Error near line 20: derivative (VFIT, mV) = 1
Derivative of non-transformation equation [BV]
** Error near line 21: derivative (VFIT, b1) = 1
Derivative with respect of non-parameter [mV]
** Warning near line 22: derivative (BFIT, b1) = mV + VFIT
Derivative with respect to unappropiate parameter [b1]
** Error near line 24: weight (mV) = 1
Weight of undefined equation [ZFIT]
** Error near line 25: error (BV) = mV + VFIT min = VFIT max = BFIT
Weight of non transformation or set equation [mV]
** Error near line 25: error (BV) = mV + VFIT min = VFIT max = BFIT
Illegal identifier in error equation [VFIT]
** Error near line 25: error (BV) = mV + VFIT min = VFIT max = BFIT
Illegal identifier in min or max equation [VFIT]
** Error near line 25: error (BV) = mV + VFIT min = VFIT max = BFIT
Illegal identifier in min or max equation [BFIT]
** Error near line 28: plot (BV) = V, B
Plot of undefined equation [QFIT]
** Error near line 29: plot (BFIT) = VFIT + BV, b1
Plot of non-transformation equation [BV]

** End of compilation **

Catalog input variables         = 2
First catalog column            = INDEF
Last catalog column             = INDEF

Observational input variables   = 3
First observational column      = INDEF
Last observational column       = INDEF

Fitting parameters              = 3
Constant parameters             = 0

Auxiliary (set) equations       = 2
Transformation equations        = 2

Warnings                        = 1
Errors                          = 19
