---
title: "harwood: where this came from and where it is"
subtitle: "A modern replacement for IRAF — project record"
date: "18 November 2026"
mainfont: "Linux Libertine O"
monofont: "DejaVu Sans Mono"
monofontoptions: "Scale=0.82"
fontsize: 11pt
papersize: a4
geometry: "margin=2.6cm"
toc: true
toc-title: "Contents"
colorlinks: true
linkcolor: black
urlcolor: "[RGB]{154,91,0}"
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\small harwood}
  - \fancyhead[R]{\small 17 November 2026}
  - \fancyfoot[R]{\small\thepage}
  - \renewcommand{\headrulewidth}{0pt}
  - \usepackage{booktabs}
  - \usepackage{longtable}
---

<!--
  Source of ~/Documents/"harwood - registro do projeto.pdf". Rebuild with
      python docs/source/build_docs.py registro
  The previous edition (13 September 2026) is kept in docs/historico/.
-->

This is the state of the project, how it got here, and what is left. It is meant to be
readable by someone who has not seen the code — a supervisor, a collaborator, or the author in
six months. The first edition of this record was written on 13 September 2026, the day after
the rebuild began; this one replaces it.

# The idea

IRAF is forty-five years old and still the only environment that treats a reduction as one
coherent system rather than a pile of scripts: tasks with persistent parameters, an
interactive display wired to the analysis, an auditable trail written into every file. It is
also dated, steep, and increasingly hard to install.

The algorithms have largely been ported to Python — ccdproc, photutils, specreduce,
astroscrappy. What had not been ported is the environment around them. An astronomer today
assembles ad-hoc scripts and loses the thread of what they did.

harwood is that environment, rebuilt. The constraint that shapes everything is migration:
task names, parameter names and muscle memory are kept, so that someone with twenty years of
IRAF in their fingers can move without relearning the vocabulary. `phot`, not
`aperture_photometry`. `apertures`, `annulus`, `dannulus`, `zmag`, `epadu`. `order` meaning
number of terms, not degree.

Two things were added to that constraint as the work went on, and they now define the project
as much as the first one:

- **The numbers, not just the names.** A task called `imcombine` that rejects differently from
  IRAF's is a trap for exactly the people it is meant to serve. Every task that has an IRAF
  counterpart is checked against IRAF itself, number for number.
- **Fidelity is the floor, not the ceiling.** In the author's words: "não é pra ser só uma
  repaginada, um IRAF mais bonito — quero que ele consiga ir onde o IRAF não consegue." Where
  a current method exists that IRAF does not have, it is welcome — as an option or a task of
  its own, never by changing what IRAF's defaults give.

The name is Margaret Harwood (1885–1979), the first woman to direct an independent
observatory, who spent forty years measuring fluxes carefully and recording how each
measurement was obtained.

# The prototype, and what it was for

The first version — about 1,850 lines across fifteen files — set out to be a shell,
explicitly: "não reimplementa algoritmo nenhum: orquestra, valida e registra." It wrapped
ccdproc and photutils, and it had the shape of the idea already: a Pydantic-based Task, a
registry, a JSONL provenance log, and a YAML recipe runner. Three of its decisions survive
intact: Pydantic models as "the modern `.par` file", provenance as a first-class object, and
the registry.

Three did not. `harwood run daofind -s input=x` became `harwood daofind x`, a POSIX command
line generated from the model. Parameters persisted to the home directory became a
`harwood.yaml` in the reduction directory — IRAF's `uparm$` problem, where a value set three
months ago elsewhere is still in force and invisible, is not repeated. And the pure wrapper
became harwood's own numerical core, for the reason the next section gives.

## Why it stopped being a wrapper

Fidelity to IRAF turned out to mean fidelity to IRAF's numbers, and a wrapper cannot promise
that. `imcombine`'s rejection centres its clipping on the median (`mclip`), as IRAF does;
without it one strong outlier drags the mean and the clip rejects the whole stack. IRAF's
apphot error model uses the measured sky scatter; a theoretical Poisson sky understated the
error of faint stars by a factor of seven on the M92 data — measured, not argued. And `icfit`
is shared infrastructure: the same fitting, rejection and keystrokes in the overscan, the
trace, the dispersion solution and the sensitivity function. That sharing *is* the
environment, and it cannot be assembled from four libraries that each fit differently.

# The two documents

Before the rebuild, two documents were written, and they still govern it:

- **`docs/iraf-inventory.pdf`** — every task of Community IRAF 2.18.2, sorted into domains,
  with what makes IRAF a reference and where it trips. Its machine-readable half is
  `src/harwood/data/iraf_migration.yaml`: 614 task names across 53 packages, each with a
  status. `harwood list --iraf` reads it, `harwood help <iraf-task>` answers from it for a
  task harwood does not have, and the test suite fails if its counts drift.
- **`docs/design-decisions.pdf`** — the contract. Sections 11 (modern instrumentation) and 12
  (SCI/VAR/DQ) were added on 21 October; sections 9 and 10, the state of the code and what is
  open, were revised on 17 November alongside this record.

The rule since: when an implementation question comes up, the answer should be in the
contract; when it is not, it gets added there before it gets written in code.

# What runs

**147 tasks in 23 packages**, under IRAF's own package and task names. Against the migration
map, of IRAF's 614 names:

| status | count | meaning |
|:--|--:|:--|
| available | 135 | shipped and complete |
| replaced | 94 | the ecosystem already does it; the map says with what |
| wontport | 162 | deliberately not portable: tape drives, old hardware, IRAF's own internals |
| planned | 223 | will exist; the map says when |

All 57 tasks of the MVP run. Spectroscopy, which the scope decision of 12 September left
*outside* the MVP because it depends on `icfit` and the `database/`, came in afterwards and is
complete through extraction, wavelength calibration, flux calibration and long-slit
rectification.

| IRAF package | available in harwood |
|:--|:--|
| images.imutil | hedit hselect imarith imcopy imdelete imexpr imgets imheader imhistogram imrename imreplace imstatistics imsum listpixels minmax sections |
| images.immatch, imgeom, imfilter, imfit | imcombine xyxymatch imcentroid imalign imshift boxcar median fmedian fit1d |
| images.tv | display imexamine tvmark wcslab |
| imred.ccdred (+ ccdtest) | ccdproc combine zerocombine darkcombine flatcombine ccdmask badpiximage ccdlist ccdgroups ccdhedit ccdinstrument setinstrument, and artobs mkimage subsection |
| imred.crutil, generic | cosmicrays background |
| digiphot.apphot | daofind phot qphot center fitsky radprof |
| digiphot.photcal | mkcatalog mkconfig chkconfig mkimsets mknobsfile mkobsfile mkphotcors obsfile mkapfile fitparams evalfit invertfit |
| twodspec.apextract | apall apfind apedit aprecenter apresize aptrace apsum |
| twodspec.longslit | response illumination fitcoords transform |
| onedspec | identify reidentify autoidentify dispcor disptrans refspectra standard sensfunc calibrate splot specplot scombine sarith scopy fitprofs continuum telluric skytweak deredden dopcor … (35 in all) |
| astutil, noao | setairmass setjd airmass asttimes precess rvcorrect observatory |
| plot, proto, artdata | implot graph contour surface; fixpix mskexpr text2mask imextensions mimstatistics; mkobjects mknoise mkpattern starlist |

Two external packages that the course uses alongside IRAF are in as well: **RVSAO**'s `emsao`
and `xcsao`, the Smithsonian's redshift tasks, checked against RVSAO 2.8.5.

## Beyond IRAF

Fifteen tasks have no IRAF counterpart and are listed apart in the map (`added:`), because the
IRAF side must stay an honest inventory of IRAF — someone migrating reads it to learn what they
lose:

| task | what it is |
|:--|:--|
| `psfphot` | PSF photometry with an empirical PSF (Anderson & King) |
| `zogy` | image subtraction with the proper statistic (Zackay, Ofek & Gal-Yam 2016) |
| `coadd` | the optimal co-addition (Zackay & Ofek 2017) |
| `drizzle` | Fruchter & Hook resampling |
| `skymodel` | the SExtractor-style background mesh |
| `skyspline`, `flexure` | Kelson's sky subtraction on the slit; flexure from the sky lines |
| `platesolve` | blind astrometry against Gaia |
| `lombscargle`, `bls` | periods and transits |
| `zfind` | redshift from all lines at once, a χ² scan as DESI's Redrock does |
| `lineid` | names the features a spectrum shows |
| `impicture` | an image to PNG or JPEG for a report |

Methods without an IRAF to check against are checked against known truth or against the
published reference implementation: L.A.Cosmic against astroscrappy (325 against 325 pixels on
the 1992 spectra), the ePSF against photutils. Where a method could not be shown on real data,
the documentation says so — `platesolve` has not yet solved a real frame of the course, and
the task refuses rather than return a guess.

## The window

`harwood gui` is a desktop application of its own (PySide6 and pyqtgraph; no DS9). The image
in one tab, and every graph a task opens in a tab beside it: `implot`, `splot`, `specplot`,
`icfit`, `identify`, `apedit`, `fitcoords`' `igsfit`, the photcal and ccdred graphs. Each one
takes IRAF's keys and colon commands, and its graphs are checked against IRAF's metacode.

Around that: a package browser that walks IRAF's tree with a breadcrumb back (no `bye`, no
reloading packages); a form for any task, generated from its model, that always shows the
equivalent command; a command line with history, completion and chaining; an editor for text
files; the documentation inside the app. The session — images, graphs, open files — comes
back when the window is opened again, if wanted. A crosshair and cursor helpers, as IRAF's
graphics cursor. Figures IRAF wrote as `.eps` open in a tab, to compare with.

Accessibility is not a mode bolted on: light and dark themes, a colour-blind-safe palette
(Okabe–Ito), a dyslexia mode, text that scales everything with one number. The interface, the
help, the warnings and the explanations are in English, Portuguese and Spanish; task names,
parameters, keys and what IRAF itself prints are never translated.

# What makes it different from a pile of scripts

## Four layers of transparency

The criticism that shaped this most was a supervisor's: that a pipeline does everything "under
the hood" and spits out a result nobody can interrogate. The answer is four layers, at four
different times:

| when | what |
|:--|:--|
| before | `--plan` — the steps, read from the real headers, including the ones that will be skipped |
| during | `--explain` — each step's reasoning, the physical concept, the numbers measured before and after |
| after | `harwood log`, `log --full`, `log --replay N` — every parameter with where it came from, and the command that repeats a run exactly |
| in the data | HISTORY cards, and OVERSCAN/TRIM/ZEROCOR marks that stop a frame being corrected twice |

## Every notice says what to do

A warning that only says what happened leaves the reader where they were. Every notice carries
a suggestion, and a test walks the source and fails on any notice raised without one:

```
▲ warning  only 5 different lines along the dispersion, for yorder 6: the
           surface passes through them and is free between and beyond them
  -> try: identify more lines in the arc (in identify, l finds the rest of
          the line list), reidentify and run fitcoords again
```

Colour is never the only channel: the level is a word and a symbol as well as a colour.

## It refuses rather than invents

Several quite different dispersions each explain a couple of dozen lines of a dense arc list,
and a cubic fits any of them to a tenth of a pixel. `identify` therefore accepts a searched
solution only when it beats every rival at a materially different dispersion. The same
discipline is in `lineid`, `platesolve` and `zfind` (which reports a second redshift whenever
the χ² cannot tell them apart).

## It never breaks the IRAF user's files

Images are written as SCI/VAR/DQ — the data, its variance, and the quality of each pixel — so
that uncertainties propagate through every step instead of being lost. But nobody is forced
off IRAF to use it: `--format iraf` writes one HDU with the mask beside it, as IRAF reads, and
every reader accepts IRAF's files, `.pl` masks included.

## It does not need IRAF

The IRAF tables the tasks read (line lists, standard stars, extinction curves, instrument
translations) ship inside the package. harwood works on a machine where IRAF was never
installed; IRAF is only ever the answer key during development.

# How it was checked

The distinctive claim of this project is that the numbers are right, so the checks are worth
listing as results.

**Against IRAF itself.** IRAF 2.18 and PyRAF run on the development machine. For each task,
IRAF's output on a set of cases is frozen in `tests/data`, and harwood is compared with it
number for number. When the manual and IRAF's code disagree — `imexamine`'s FWHM came out 5%
off by following the manual — the SPP source is ported line by line. Some results:

| check | result |
|:--|:--|
| imshift | bit for bit |
| imcombine, all rejections, scale/zero/weight, grow | IRAF's numbers |
| photcal: fitparams, invertfit, mkapfile, obsfile, mkimsets | identical, byte for byte in the files |
| phot on 12 Skinakas frames | within half a unit of the last printed digit |
| cosmicrays `--algorithm iraf` | pixel, mask and CRCOR as IRAF's, up to 31,669 replacements on M92 |
| contour and surface | IRAF's NCAR curves and hidden lines, against its metacode |
| emsao (RVSAO) | the emsao.log line for line, 44 cases |
| xcsao (RVSAO) | the xcsao.log line for line, 135 cases (every report mode; correlation in velocity, wavelength or pixels); the header cards `save_vel` writes, card for card |
| 1992 airmasses | all seven object frames to 0.01, from the site table |

**Against the course.** Every exercise of the IRAF course (introduction, CL and display, CCD
reduction and photometry of M92, the Skinakas photometry and its calibration, spectra from
zero and flat to wavelength calibration, and the long-slit NGC 660 from raw frames to its
redshift) has a walkthrough that runs in harwood and is checked against the results obtained
in class with IRAF. On NGC 660, with the class's own sky sample, `background` gives the class's
sky-subtracted image pixel for pixel; the nucleus' Hα and [N II] centres agree with the class's
`splot` to 0.1 Å; `emsao` gives cz = 869 ± 13 km/s from five lines.

**Against physics,** where no reference existed: the heliocentric correction is bounded by the
light-crossing time of the Earth's orbit; the [O III] 4959/5007 ratio is fixed at 3:1 by
atomic physics; a flux calibration must recover a second star at a different airmass.

**Against known truth,** for everything that integrates or calibrates: invent the answer,
synthesise the observation, demand the truth back. That is how a 4% bias in bandpass
integration and an optimal extraction that was a no-op on the flux were found — both passed
every test that compared a measurement with its own error bar.

**7,167 tests**, on Linux and Windows, Python 3.12 and 3.14.

## Bugs found in IRAF

Porting line by line also finds IRAF's own mistakes. Two that harwood does not reproduce, on
purpose and with a test that says so: `apall` with variance weights and a gain other than one
divides only the first subaperture by the gain; and the Marsh `fit2d` profile with its default
order is so ill-conditioned in single precision that IRAF itself is not reproducible. The
course material has its own: `imreplace lower=0.` replaces the pixels *at or above* zero, not
the negative ones the text says — the recipe works because of that. And RVSAO's `save_vel`
writes back into the file the whole header it holds in memory, with the 00:00:00 RA and Dec
it made up while reading a spectrum that had none and today's date: the next task — the next
aperture of the same multispec, to begin with — then corrects the velocity for an object at
0h 0° observed today. harwood writes RVSAO's own cards and nothing else.

## What the course taught the software

Running the exercises the way a student does found what tests did not. This week alone:
`background --interactive` started from the window asked its question at a terminal the
window does not have, and finished without opening the fit; and an `identify` saved with only
the five lines marked by hand looked perfect (RMS 0) and quietly gave `fitcoords` a surface
600 Å wrong at the ends. Both now say what is happening — the second one where IRAF says
nothing at all.

# What is left

**Next.** The rest of RVSAO: the `emsao` and `xcsao` graph with its cursor keys (xcsao's
default display is emsao's), `emsao` saving its velocity to the header and its sky spectrum,
then `fxcor`, which was deferred for its size. `xcsao` itself is otherwise complete: its
correlation in velocity, wavelength or pixels, every report mode, and `save_vel`.

**Planned.** 223 IRAF tasks remain on the map, most of them later milestones: `daophot`
(13), `imcoords` (16), the rest of `immatch` (`geomap`, `geotran`, `psfmatch`, mosaics),
`quadred`, `echelle` and the fibre pipelines.

**Modern detectors.** Section 11 of the contract: multi-amplifier CCDs and mosaics, CMOS,
EMCCD and infrared arrays, each behind a switch that leaves IRAF's results untouched by
default. None of it is written before a real data set of that kind exists to exercise it.

**Open decisions.** The compatibility policy between versions; the first release on PyPI; the
`rust` extra, built but not yet published.

## Known limitations, stated plainly

- `platesolve` has not solved a real frame from the course: the M92 tutorial frames give
  neither scale nor orientation, and the 1987 pointing looks wrong. The task refuses rather
  than guess.
- No flux standard was observed on the 1992 spectroscopic night, so the flux chain cannot be
  shown on it; it is verified in closed loop against IRAF's own standard-star tables, with a
  second star at another airmass.
- `fxcor` does not exist yet; `xcsao` covers the cross-correlation the course uses.
