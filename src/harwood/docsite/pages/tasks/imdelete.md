Deletes images. By default it does **not**: it says what it would delete
and stops, because a pattern that matches more than you meant is how a
night's work disappears. Read the list, then run it again with
`--no-verify`.

```example
title: What would go
after: reduced
run: harwood imdelete 'red/obj00*.fits'
fails: true
output: |
  3 image(s) would be deleted: obj001.fits, obj002.fits, obj003.fits
```

`harwood sections 'red/obj00*.fits'` shows what a pattern matches without
the question.
