Fits the transformation equations -- how an instrumental magnitude depends
on the standard magnitude, the colour and the airmass -- to the standard
stars of the night, and appends the coefficients to a parameters file that
[invertfit](task:invertfit) reads.

```console
$ harwood fitparams observations.obsfile catalogo.mkcatalog configuration.mkconfig fitfile.fitparams --logfile fitparams.log --log-fit --log-results
```

**The configuration is the model.** Nothing about a photometric system is
built in. The configuration file, in IRAF's language, says which column of
the catalog and of the observations holds which variable, and writes each
equation, for instance

```text
transformation
fit   v1 = 25.0, v2 = -0.03, v3 = 0.18
VFIT : mV = V + v1 + v2 * (B - V) + v3 * XV
```

`const` fixes a parameter, `delta` sets the step of its numerical
derivative, `set` names an expression to use in the others, and
`derivative`, `error` and `weight` equations do what IRAF's do. The same
file works in IRAF and in harwood, both ways. [mkconfig](task:mkconfig) makes one
from a standards catalog's format file and a template of equations, and
[chkconfig](task:chkconfig) says what is wrong with one, line by line.

**Matching.** Each observation is paired with its star in the catalog by
the id in the first column, upper-cased and reduced to letters, digits, `+`,
`-` and `_`. Observations with no catalog line -- the program stars, or a
standard misspelt -- take no part, and are listed.

**The fit.** Each equation is fitted alone by IRAF's nlfit
(Levenberg-Marquardt) from the initial values in the configuration.
`--weighting uniform` counts every star the same; `photometric` by the
inverse variance of the magnitudes in the equation, and `--addscatter` fits
an extra scatter term so that stars with underestimated errors do not
dominate; `equations` uses the configuration's weight equations.
`--nreject` passes of rejection at `--low-reject`/`--high-reject` times the
rms refit from the initial values each time; they need five stars.

**By the keys.** `--interactive` opens each equation's fit in a tab beside the
image; `--gcommands` takes the same keys from a file, `x y wcs key` a line, and
`--answers` the lines the prompts read, without a screen.

```console
$ harwood fitparams observations.obsfile catalogo.mkcatalog configuration.mkconfig fitfile.fitparams --interactive
```

`h`, `i`, `j`, `k` and `l` are five graphs, function against fit, residuals and
ratio, the first variable against residuals, and the configuration's `plot`
equations when it has them, which are then what opens. `g` says what one of them
draws: `function`, `fit`, `residuals`, `ratio`, `nonlinear`, `var 11`, `user 1`,
or a variable's name, as `XV,residuals`. Deleted stars show as ×, rejected ones
in a diamond, and `t` adds the fit's own values. `c` names the star nearest the
cursor with its function, fit and residual.

`d` deletes a star and `u` brings it back, but the fit changes only with `f`.
The colon commands change the next `f` too, not what is on the screen:
`:const v2` holds a parameter at its value, `:const v2 0.1` at that one, and
`:fit v2` frees it, by name or number; `:nreject`, `:low_reject`, `:high_reject`,
`:grow`, `:tolerance` and `:maxiter` take a value, or show it. `:errors`,
`:results`, `:vshow`, `:show`, `:variables` and `:data` print what they name,
into a file if you give one.

`q` leaves the equation. The fit is not redone: leave with a star deleted and no
`f`, and the errors saved are those of the fit you had, over the stars as they
now stand. The panel asks whether to save, and where to go: `next`, `prev`,
`first`, `last`, `again` for the same equation, `go` to save the rest unasked,
`quit`. An empty answer is the default the prompt names. A fit that converged and
was saved is what that equation starts from if you come back to it, and an
equation that refers to it sees its values; every record saved is appended, and
invertfit reads the last. What `:const` and `g` changed lasts for that turn of
the equation. Closing the tab before the end saves nothing more, and the task
says which equations have no record.

**Identical to IRAF.** The parameters file and the log agree with IRAF's
to the last digit printed, in every weighting, with rejection, and with the
whole configuration language exercised; so do thirty-three sessions by the keys,
with the text they print on both streams. Three differences are deliberate.
`:const 3` names the parameter, where IRAF prints whatever was in memory. `w` and
the graphics cursor's `:.` and `:/` commands are not ported: the mouse zooms the
panel. `?` lists the keys above instead of paging IRAF's help file. One IRAF
habit is kept for the file of answers only: a blank line after a command it did
not know means `first`.
