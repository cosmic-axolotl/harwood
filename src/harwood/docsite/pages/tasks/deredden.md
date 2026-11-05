Corrects spectra for interstellar extinction, by Cardelli, Clayton & Mathis's
law (1989, ApJ 345, 245).

```console
$ harwood deredden hd37022.fits hd37022.d.fits 0.34
[hd37022.d]: HD 37022
  Nov  5  7:29 E(B-V)=0.34 A(V)=1.054 R=3.1
  Ap 1: HD 37022
$ harwood deredden pn.fits pn.d.fits 0.62 --type c --R 3.1
```

The value is the extinction, as `--type` says: `E(B-V)` (the default), `A(V)`,
or `c`, the logarithmic extinction at H-beta that nebular work measures from
the Balmer decrement, A(V) = R c (0.61 + 0.024 c). `--R` is A(V)/E(B-V), 3.1
for the Galaxy's diffuse medium and larger in dense clouds. Every pixel of the
apertures chosen (`--apertures`, all by default) is multiplied by
10^(0.4 A(V) A(λ)/A(V)), in every band; the law reaches from 1000 to 33333 Å,
and a spectrum beyond that is an error, as in IRAF.

The header gets a `DEREDDEN` card with the date and what was done. A spectrum
that has one is not corrected twice: `--override` corrects it again, and takes
the old correction out first -- its A(V) and R read back from the card --
unless `--no-uncorrect` says to leave it in.

Everything is computed in real, as IRAF does it: nine sessions IRAF 2.18 ran
-- each type, another R, some apertures, a second correction refused,
overridden with another R and with the same one, the old one kept, and a
spectrum too far in the ultraviolet -- say and write the same, and of 600
pixels per case all but a few are IRAF's to the bit, those few one unit of the
last place off. IRAF corrects in place when given no output; harwood writes to
the run's directory, and the input's own name as the output corrects it in
place.
