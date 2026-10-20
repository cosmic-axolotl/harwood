Gets one keyword from one header, for a script to read: the value, and
whether it was there at all. `--default` is what comes back when it is not.

```example
title: The exposure time of a frame
run: harwood imgets obj001.fits EXPTIME
output: |
  value: 60.0
  found: True
```

From Python, `harwood.run("imgets", image="obj001.fits", param="EXPTIME")`
gives the same in `result.data["value"]`.
