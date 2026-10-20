# Notices: what happened, and what to do

A warning that only says what happened leaves you where you were. Every
notice harwood raises has two halves: what happened, and what to try next.

```text
▲ warning  1 star: the centre moved more than maxshift from where it started (cier 107)
   -> try: look at the flagged rows (cier) in the table; for coordinates taken
           from another frame raise --maxshift, and for faint stars lower --minsnratio
```

## Three levels

| | level | means |
|---|---|---|
| ● | **note** | worth knowing; the result is fine |
| ▲ | **warning** | the result is usable, but something about it should be checked |
| ✖ | **problem** | the result is probably wrong -- it is still given, because a wrong answer with the reason beats no answer |

**Colour is never the only sign.** Each level is written as a word and has
its own symbol, so it survives a pipe into a file, a screen reader, a
printout in grey, and colour blindness. The colours themselves are Okabe
and Ito's, which stay apart in the common colour-vision deficiencies.

## When a task stops

A task that cannot go on says so in a sentence, with the same two halves,
and the session or the window carries on:

```example
title: A task given a file that is not there
run: harwood imstatistics missing.fits
fails: true
output: |
  imstatistics stopped
```

`HARWOOD_TRACEBACK=1` brings back the Python traceback, for when the
sentence is not enough -- hidden by default, never unavailable.

## A name harwood does not run

An IRAF task that harwood does not have gets an answer, not *command not
found*: whether it is planned, replaced by something else, or not ported
and why. A typo gets the nearest names.

```example
title: An IRAF name
run: harwood help splot
fails: true
```
