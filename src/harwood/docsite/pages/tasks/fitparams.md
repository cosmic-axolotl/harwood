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

**The graph.** `w` and a key after it set the window, as IRAF's gtools do: `z`,
`x` and `y` zoom about the cursor, `e` and `e` again mark two corners, `j` `k`
`b` `t` an edge, `l` `r` `u` `d` shift, `c` and `p` centre and pan, `f` and `g`
flip an axis, `a`, `m` and `n` let the limits go, `w ?` lists them. The colon
commands that begin with `/` are gtools' too: `:/xwindow -8 -6` and `:/ywindow`
set the window or show it, `:/expand` and `:/shift` move it, `:/xflip yes`,
`:/transpose`, `:/xtransform log` (IRAF's elog scale, out to whole decades),
`:/mark box`, `:/type line`, `:/xsize`, `:/subtitle`, `:/comments`,
`:/drawtitle no`, `:/drawxlabels no`, and `:/help` for the rest. As IRAF's help
says, they take effect at the next graph (`r`), and the title, the labels and
the parameters line are set anew at each one, so `:/title` changes nothing here.
The window starts over with each equation; flips, marks and the rest carry on to
the next. On a flipped axis an edge is the screen's: `w j` is the left one
whichever way the numbers run. The mouse zooms and drags the graph besides, and
`c`, `d` and `u` take the point nearest on the screen as it is. `:.` commands
are IRAF's cursor mode, which the panel has in part: `:.snap graph.png` (or .svg,
.pdf, .csv) saves the graph, `:.help` says what else there is.

**Identical to IRAF.** The parameters file and the log agree with IRAF's
to the last digit printed, in every weighting, with rejection, and with the
whole configuration language exercised; so do forty-six sessions by the keys,
with the text they print on both streams -- `w` with every key after it, the
`:/` commands, flipped, transposed and logarithmic graphs among them. What is
drawn is held to IRAF's graphics metacode as well: 223 frames, windows, titles,
labels and every mark the window leaves whole. Three differences are deliberate.
`:const 3` names the parameter, where IRAF prints whatever was in memory. `?`
lists the keys above instead of paging IRAF's help file. Cursor mode's `:.`
commands are the panel's, as they are the terminal's in IRAF; from a file of
cursor commands they reach the task, which rings the bell, as IRAF's does. One
IRAF habit is kept for the file of answers only: a blank line after a command it
did not know means `first`. Not ported: `I`, which kills the task.
