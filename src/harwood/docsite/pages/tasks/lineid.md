Names the features of a wavelength-calibrated spectrum: which atom, ion or
molecule each line belongs to, from a curated catalogue of the lines that
matter in stars, nebulae and the Earth's own atmosphere. A complete list
would match everything and so tell nothing; this one is kept short on
purpose.

```console
$ harwood lineid obj.w.fits --features absorption --plot
```

- Features are detected above `--threshold` times the local noise
  (absorption, emission, or both) and matched to catalogue lines within
  `--tolerance` Å, after moving the catalogue to the object's
  `--velocity`.
- Each identification says the species, the ion and the band, and which
  are ambiguous -- two catalogue lines within the tolerance of one feature.
- From the lines identified it works out whether they agree on a radial
  velocity, and says what velocity that is.
- `--origin` keeps one kind (stellar, nebular, telluric…); `--prominence`
  drops the fainter catalogue lines.
- `--plot` opens the spectrum with every catalogued line drawn where it
  falls, coloured by what produces it -- and named, not only coloured -- so
  that an absorption that is oxygen above the telescope does not pass for
  one in the star.

The same catalogue answers from the terminal: `harwood lines 6562.8` says
what might be at a wavelength, `harwood lines "H I"` lists a species, and
`harwood lines --range 6700 6720` shows a neighbourhood.
