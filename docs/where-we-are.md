# harwood: where this came from and where it is

*Written 13 September 2026.*

This is the state of the project, how it got here, and what is left. It is
meant to be readable by someone who has not seen the code — a supervisor, a
collaborator, or the author in six months.

---

## 1. The idea

IRAF is forty-five years old and still the only environment that treats a
reduction as one coherent system rather than a pile of scripts: tasks with
persistent parameters, an interactive display wired to the analysis, an
auditable trail written into every file. It is also dated, steep, and
increasingly hard to install.

The algorithms have already been ported to Python — `ccdproc`, `photutils`,
`specreduce`, `astroscrappy`. What has not been ported is the **environment
around them**. An astronomer today assembles ad-hoc scripts and loses the
thread of what they did.

harwood is that environment, rebuilt. The constraint that shapes everything is
migration: task names, parameter names and muscle memory are kept, so that
someone with twenty years of IRAF in their fingers can move without relearning
the vocabulary. `phot`, not `aperture_photometry`. `apertures`, `annulus`,
`dannulus`, `zmag`, `epadu`. `order` meaning number of terms, not degree.

The name is Margaret Harwood (1885–1979), the first woman to direct an
independent observatory, who spent forty years measuring fluxes carefully and
recording how each measurement was obtained.

## 2. The prototype, and what it was for

The first version — about 1,850 lines across fifteen files — set out to be a
**shell**, explicitly: *"não reimplementa algoritmo nenhum: orquestra, valida e
registra."* It wrapped `ccdproc` and `photutils`, and it had the shape of the
idea already:

```
harwood run daofind -s input=ngc6752.fits -s output=cat.ecsv
harwood epar daofind -s fwhm=3.8
harwood recipe run recipes/photometry_basic.yaml
harwood recipe from-log -o minha_pipeline.yaml
```

Two task modules, a Pydantic-based `Task`, a registry, a JSONL provenance log,
and a YAML recipe runner. It proved the architecture was viable and it
established the vocabulary. Three of its decisions survive intact: Pydantic
models as "the modern `.par` file", provenance as a first-class object, and the
registry.

Three of its decisions did not, and the reasons are the interesting part:

| prototype | now | why it changed |
|---|---|---|
| `harwood run daofind -s input=x` | `harwood daofind x` | A set/get syntax is a second language on top of the shell. §3.1 of the contract fixes a POSIX command line, generated from the Pydantic model, so the task list and the CLI cannot disagree. |
| `epar` persisting parameters to the home directory | `harwood.yaml` in the reduction directory | This is IRAF's `uparm` problem: a parameter set three months ago in another directory is still in force and invisible. Parameters now live *with the reduction*, and nothing is written to `$HOME`. |
| a pure wrapper around ccdproc/photutils | harwood's own kernels for the numerical core | See below. This is the single largest change. |

### Why it stopped being a wrapper

Fidelity to IRAF turned out to mean fidelity to IRAF's *numbers*, not just its
names, and that is not something a wrapper can promise. Concretely:

- `imcombine`'s rejection defaults to **median-centred** clipping (`mclip`), as
  IRAF does. Without it, one strong outlier drags the mean and the clip rejects
  the whole stack. A wrapper inherits whatever its dependency chose.
- IRAF's `apphot` error model is a three-term formula using the **measured**
  sky `stdev`. A theoretical Poisson sky understated the scatter of faint stars
  by a factor of seven on the M92 data — measured, not argued.
- `icfit` is shared infrastructure: the same fitting, rejection and keystrokes
  in the overscan, the trace, the dispersion solution and the sensitivity
  function. That sharing *is* the environment, and it cannot be assembled from
  four libraries that each fit differently.

## 3. The two documents

Before the rebuild, two documents were written:

- **`docs/iraf-inventory.pdf`** — all 610 task names of Community IRAF 2.18.2
  across 52 packages, sorted into 24 domains, with what makes IRAF a reference
  and where it trips. Its machine-readable half is
  `src/harwood/data/iraf_migration.yaml`, which `harwood list --iraf` reads and
  which the test suite checks for drift.
- **`docs/design-decisions.pdf`** — the contract. Nine sections, each of which
  now names the file that implements it (see `docs/README.md`).

The rule since: when an implementation question comes up, the answer should be
in the contract; when it is not, it gets added there before it gets written in
code.

## 4. The rebuild

On 12 September 2026 the decision was to **stop reconciling the prototype with
the contract and start again from the contract**. The old repository remains at
`~/Desktop/harwood` as a historical reference and is not edited.

Twenty-one commits later, the shape is:

```
src/harwood/
  core/      30 modules -- the arithmetic, none of it task-specific
  kernels/   numba and Rust backends, dispatched at run time
  tasks/     the IRAF tasks, grouped by IRAF package
  viewer/    the embedded PySide6 + pyqtgraph window
  cli/       the POSIX command line, generated from the models
  data/      the migration map, the spectral line catalogue
tests/       22 files
recipes/     worked reductions on real data
```

About 17,000 lines of source and 5,800 of tests.

## 5. What runs

**36 tasks**, in IRAF's own package names:

| package | tasks |
|---|---|
| `imutil` | `hedit` `hselect` `imarith` `imcopy` `imdelete` `imgets` `imheader` `imstatistics` `sections` |
| `ccdred` | `ccdproc` `ccdlist` `setinstrument` `zerocombine` `darkcombine` `flatcombine` |
| `immatch` | `imcombine` `xyxymatch` `imcentroid` `imalign` |
| `apphot` | `daofind` `phot` |
| `apextract` | `apall` (with `apfind`, `apedit`, `apresize`, `apsum`) |
| `onedspec` | `identify` `refspectra` `dispcor` `standard` `sensfunc` `calibrate` `wspectext` `lineid` |
| `astutil` | `setairmass` `setjd` `airmass` `asttimes` `observatory` |
| `plot` | `implot` |

Against the migration map: **24 of the 57 MVP tasks**, plus 15 more that were
scheduled for later and were pulled forward because the spectroscopic chain
needed them. Of the 610 IRAF names, 94 are marked *replaced* (the ecosystem
already does it better) and 161 *wontport* (tape drives, legacy hardware,
IRAF's own internals).

`lineid` has no IRAF counterpart and is listed separately in the map, under
`added:` — the IRAF side must stay an honest inventory of IRAF, because someone
migrating reads it to learn what they lose.

### The two complete chains

**Photometry.** `ccdproc` → `zerocombine`/`flatcombine` → `daofind` → `phot`,
with `xyxymatch`/`imcentroid`/`imalign` for putting frames on one grid.

**Spectroscopy.** `ccdproc` → `apall` → `setairmass` → `refspectra` →
`identify` → `dispcor` → `standard` → `sensfunc` → `calibrate`, with `lineid`
naming what the features are.

## 6. What makes it different from a pile of scripts

### Four layers of transparency

The criticism that shaped this most was a supervisor's: that a pipeline does
everything "under the hood" and spits out a result nobody can interrogate. The
answer is four layers, at four different times:

| when | what |
|---|---|
| before | `--plan` — the steps, read from the real headers, including the ones that will be skipped |
| during | `--explain` — each step's reasoning, the physical concept, the numbers measured, before and after |
| after | `harwood log`, `log --full`, `log --replay N` — every parameter, and the command that repeats a run exactly |
| in the data | `HISTORY` cards, and `OVERSCAN`/`TRIM`/`ZEROCOR` marks that stop a frame being corrected twice |

### No hidden state

Parameters resolve through four levels — command line, project file, preset,
model default — and every value can say which level it came from (`harwood
lpar`). `harwood.yaml` lives in the reduction directory. Nothing is written to
the home directory, ever.

### Every notice says what to do about it

A warning that only says what happened leaves the reader where they were. All
of them now carry a suggestion:

```
▲ warning  50 of 2100 pixels fall outside the 3350 to 7450 A the standards
   covered. Their calibration is extrapolated, and a response extrapolated
   past its last band goes wrong quickly.
   -> try: trust the spectrum only between 3350 and 7450 A, or run standard
           on a calibration star whose table reaches further.
```

Colour is never the only channel: the level is spelled out in words, each level
has its own symbol, and the palette is Okabe-Ito, so the severity survives
greyscale, a pipe into a file, a screen reader, and colour blindness. A test
walks the source and fails on any notice raised without a suggestion.

### It refuses rather than invents

The clearest example is the dispersion solution. Against a thorium list with
one line every 2.5 Å, several quite different dispersions each explain a couple
of dozen arc lines, and a cubic fits any of them to a tenth of a pixel. The fit
looks excellent and the answer is invented. `identify` therefore accepts a
searched solution only when it beats every rival at a materially different
dispersion, and refuses to write a solution whose scatter exceeds what a line
centre can be measured to. The same discipline appears in `lineid` (two species
within the tolerance are both reported, and the identification is called
ambiguous) and in `search_dispersion`.

## 7. How it was checked

The distinctive claim of this project is that the numbers are right, so the
checks are worth listing as results rather than as test names.

**Against the course's IRAF exercises, on real data:**

| check | result |
|---|---|
| Exercise 2 flat modes | 1313.03 and 1468.73, against the text's 1313 and 1468 |
| M92 detection completeness | 98%, centroids good to 0.033 px |
| M92 photometry, two V frames | agree to −9 mmag with 3 mmag scatter |
| M92 predicted vs measured error | ratio 1.0 (was 7× wrong before the IRAF error model) |
| 1992 airmasses, all seven object frames | reproduced to better than 0.01 from the site table, RA/DEC and `25/01/92` |
| Sidereal time vs frame transformation | 1.562 by hand against 1.565, two independent routes |
| Spectral trace | 1.41 px real drift, fitted to 0.0074 px |
| Six exposures, two hours | aperture centres agree to 0.07 px |
| AH Cep repeatability | 1.2% scatter at S/N 123, matching prediction |
| `dispcor` on a synthetic arc | lines land on thorium to 0.0006 Å (0.004 px) |
| Flux calibration, closed loop | response recovered to 0.05 mag; a second star at a different airmass to 3% |

**Against physics, where no reference existed:** the heliocentric correction is
bounded by the light-crossing time of the Earth's orbit and flips sign between
January and July; the [O III] 4959/5007 ratio is fixed at 3:1 by atomic physics
and is therefore a free check on any flux calibration.

### What the tests missed, and how it was found

Three bugs survived a green test suite and were caught by running the whole
chain on real data:

1. **Optimal extraction was a no-op on the flux.** The Horne profile was taken
   from the data at that wavelength, and the estimator then collapses
   algebraically to the plain sum — bit for bit, for every input — while the
   error shrinks, because the "profile" has absorbed the noise it was meant to
   weight against. It presented as a free 17% in signal-to-noise. The test
   asserted `optimal_snr > plain_snr`, compared flux against its *own reported
   error*, and so was certifying the bug. Now the profile is fitted along the
   dispersion; the gain on bright real data is 0.3%, which is correct, and 59%
   on a faint synthetic one, measured as scatter about a known answer.
2. **`apall` wrote its trace to `database/id<image>`** — the dispersion
   solution's prefix. Run `identify` and `apall` on one frame and the second
   silently overwrites the first.
3. **A 4% bias in bandpass integration.** Integrating only over the samples
   that fall inside a band covers about one pixel less than the band; dividing
   by the *full* width then understates every band by the same fraction. It
   never shows in the rms, because every band is wrong equally.

The lesson, now written into the project's own notes: for anything that
integrates or calibrates, write a **closed-loop** test — invent the answer,
synthesise the observation, demand the truth back. A systematic error common to
every point shows up no other way. And never compare a measurement against its
own error bar.

**395 tests, ruff clean.**

## 8. What is left

### Immediately useful

- **`cosmicrays`, `ccdmask`, `fixpix`, `badpiximage`** — the 1200-second
  exposures in the exercise data certainly carry cosmic-ray hits and nothing
  removes them yet.
- **`display`, `imexamine`, `tvmark`, `rimcursor`** — the interactive display
  group. The embedded viewer exists and docks its panels; `imexamine` is IRAF's
  most-used interactive task and is the largest single gap in the experience.
- **`center`, `fitsky`, `qphot`, `radprof`** — the photometry sub-tasks. The
  chain already works; these expose its parts, which is fidelity rather than
  new capability.

### Structural

- **Runnable recipes.** The contract (§3.2) sets aside IRAF's `doslit`/`dohydra`
  pipeline tasks to become YAML recipes. `recipes/` currently holds the written
  form only.
- **Shipping the calibration data.** Line lists, standard star tables and
  extinction curves are read from an installed IRAF. §4 wants them shipped;
  their licences need checking one by one first.
- **The remaining ~80 error sites** outside the spectroscopic chain still carry
  their suggestion inline in the sentence rather than on its own line.
- **`harwood-core`** — a packaging extra that installs without Qt, for a server.

### Known limitations, stated plainly

- **The 1992 arcs' dispersion is unknown.** A search from 0.03 to 1.0 Å/px
  against every lamp list IRAF ships found nothing explaining more than 9 of the
  46 arc lines; a correct solution would explain thirty-something. This is not
  a bug — it is the shape of the problem, and it is why IRAF's `identify` is
  interactive. The way through is to mark four or five lines by hand.
- **No flux standard was observed on that night**, so the flux chain cannot be
  demonstrated on it. It is verified in closed loop against IRAF's own standard
  star tables instead.
- **`database.read_iraf`** parses IRAF's own solution files with the column
  order documented but unverified against a file IRAF actually wrote — noted in
  the code, at the place it would bite.
