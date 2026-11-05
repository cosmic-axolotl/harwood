"""The cases of IRAF's ndprep gabarito, on the CTIO filter curves harwood carries."""

CASES = {
    "linear": dict(filter_curve="nd1m.250mag.dat", w0=3500.0, dw=5.0, nw=300),
    "log": dict(filter_curve="nd4m.l250mag.dat", w0=3600.0, dw=4.0, nw=250, logarithm=True),
    "lognoflux": dict(filter_curve="nd4m.l250mag.dat", w0=3600.0, dw=4.0, nw=250,
                      logarithm=True, flux=False),
    "rows": dict(filter_curve="nd1m.100mag.dat", w0=4000.0, dw=10.0, nw=80, nspace=5,
                 dispaxis=1),
    "columns": dict(filter_curve="nd1m.100mag.dat", w0=4000.0, dw=10.0, nw=80, nspace=3,
                    dispaxis=2),
}
