Shows what an image template matches, before anything is run on it -- worth
doing first whenever a template renames things. Globs, `@lists`, comma
lists and IRAF's substitutions all work.

```example
title: What a glob matches
run: harwood sections 'obj00*.fits'
output: |
  obj001.fits
```

A substitution template is `%old%new%` followed by the rest of the name:
`%obj%rate%00*.fits` matches `obj00*.fits` and shows the names each match
would become.

```example
title: Where a template would write
run: harwood sections '%obj%rate%00*.fits'
output: |
  rate001.fits
```
