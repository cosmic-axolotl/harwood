Puts wavelengths on an extracted spectrum, from the dispersion solution of
its arc: each pixel gets the wavelength the solution puts it at.

```console
$ harwood dispcor sp0019.ms.fits sp0019.w.fits
```

**Which arc.** With `--reference`, that arc's solution. Without, the one
[refspectra](task:refspectra) wrote into the spectrum's header -- and when
it wrote two (`REFSPEC1`, `REFSPEC2`, with weights), the two solutions are
combined in that proportion, the arc before and the arc after the exposure,
so the flexure between them is followed.

**Resampling.** By default, as IRAF does, the spectrum is put on an evenly
spaced grid (`--linearize`), from `--w1` to `--w2` in steps of `--dw` or in
`--nw` pixels, or `--log` for an even grid in log wavelength. Flux is
conserved (`--flux`): each output pixel holds the counts that fall in it,
through IRAF's interpolant (`--interp`, `poly5`), so a line's equivalent
width survives -- but resampling correlates neighbouring pixels, so the noise
of a resampled spectrum looks lower than it is. `--no-linearize` keeps the
pixels and writes each one's own wavelength instead. Outside the solution's
range, `--blank` is written.

**It checks itself.** The strongest features of the result are looked up in
the line catalogue, and dispcor reports which ones landed on a real
wavelength. That is a different check from the arc's rms: the rms says the
fit reproduces the lines it was given, which a solution in entirely the
wrong place does beautifully; real features landing on real wavelengths is
what catches a solution five Ångströms out. `--no-check` skips it.

The result has a `wavelength` column beside the flux, in the same file
format as [apall](task:apall)'s; [lineid](task:lineid) names its features
and [calibrate](task:calibrate) puts it in physical units.
