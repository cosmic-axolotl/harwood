# Recipes

Worked reductions, start to finish, on real data, now part of the
documentation so that they ship with harwood and are read where everything
else is:

```console
$ harwood docs recipes/1992-spectroscopic-night
```

or *Recipes* in the documentation's contents. The source is
`src/harwood/docsite/pages/recipes/`.

The rule stays the one these began with: every command in a recipe has been
run against the data it names, and the numbers quoted are the numbers it
produced. Where a reduction cannot be finished with the data available, the
recipe says so and says what is missing, rather than stopping quietly or
inventing a step. (Last run in full on 2026-10-19, after apall, identify and
dispcor were ported from IRAF.)

| recipe | data | what it covers |
|---|---|---|
| A night of spectra, 25 January 1992 | `~/iraf/exercises/spec` | bias, extraction, tracing, arc assignment -- and the one place a human is required |

## What a recipe is not, yet

The design contract (§3.2) sets aside IRAF's pipeline tasks -- `doslit`,
`dohydra`, `dofoe` -- to be **runnable** YAML recipes rather than tasks. That
machinery is later work. What is here now is the written form: the commands,
in order, with the reasoning.
