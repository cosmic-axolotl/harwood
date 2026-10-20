Applies the sensitivity function to a wavelength-calibrated spectrum, and
takes out the atmosphere: the result is in erg/cm²/s/Å.

```console
$ harwood calibrate obj.w.fits obj.f.fits --sensitivity sens --extinction onedstds$kpnoextinct.dat
```

For each point, the counts are divided by the exposure time and by the
sensitivity at that wavelength (interpolated with `--interp`, `poly5` as in
IRAF), and multiplied by the extinction the exposure suffered at its airmass
-- from the header's `AIRMASS`, so run [setairmass](task:setairmass) first.
`--no-extinct` leaves the atmosphere in; `--no-flux` takes out only the
atmosphere.

The magnitudes of the standards' tables are turned into flux as F_ν =
3.68 × 10⁻²⁰ × 10^(−0.4 m) and F_λ = F_ν c / λ², the AB system to within
0.015 mag; BD+28 4211 at 3200 Å comes out at the published 1.515 × 10⁻¹²
erg/cm²/s/Å.

A night with no flux standard observed cannot be flux calibrated, and no
parameter changes that: see
[where the 1992 night ends](page:recipes/1992-spectroscopic-night#9-where-this-night-ends).
