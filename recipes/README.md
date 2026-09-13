# Recipes

Worked reductions, start to finish, on real data.

A task's `--help` says what one command does. These say what a *night* takes,
in what order, and why each step is there -- which is the part that is hard to
learn from a reference manual and the part an IRAF user actually needs when
deciding whether to move.

Every command here has been run against the data it names, and the numbers
quoted are the numbers it produced. Where a reduction cannot be finished with
the data available, the recipe says so and says what is missing, rather than
stopping quietly or inventing a step.

| recipe | data | what it covers |
|---|---|---|
| [1992-spectroscopic-night.md](1992-spectroscopic-night.md) | `~/iraf/exercises/spec` | bias, extraction, tracing, arc assignment -- and the one place a human is required |

## What a recipe is not, yet

The design contract (§3.2) sets aside IRAF's pipeline tasks -- `doslit`,
`dohydra`, `dofoe` -- to be **runnable** YAML recipes rather than tasks. That
machinery is later work. What is here now is the written form: the commands,
in order, with the reasoning. Running them is still your job, which for
learning a reduction is arguably the right way round.
