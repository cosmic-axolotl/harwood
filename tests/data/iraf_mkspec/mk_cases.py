"""The cases of IRAF's mkspec gabarito."""

CASES = {
    "flat": dict(image_title="flat", ncols=50, nlines=1, function=1, constant=12.5),
    "ramp": dict(image_title="ramp", ncols=37, nlines=1, function=2, start_level=100.0,
                 end_level=-3.3),
    "hot": dict(image_title="hot star", ncols=200, nlines=1, function=3, start_wave=3000.0,
                end_wave=9000.0, temperature=9500.0),
    "cool": dict(image_title="cool", ncols=120, nlines=4, function=3, start_wave=4000.0,
                 end_wave=25000.0, temperature=3300.0),
}
