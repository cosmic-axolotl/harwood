IRAF's `observatory`: the sites, as IRAF's `obsdb.dat` has them -- 74
observatories, from Kitt Peak to the LNA, which ships with harwood -- with
name, latitude, longitude (positive **west**, as IRAF stores it), height
and time zone. A header's `OBSERVAT` is looked up here, and so is
`--observatory` in every task that needs a site.

```example
title: Kitt Peak
run: harwood observatory kpno
output: |
  Kitt Peak National Observatory
```

Three commands, IRAF's:

| | |
|---|---|
| `list` (the default) | every site, or one: `harwood observatory list lna` (or just `harwood observatory lna`) |
| `set` | this reduction's default site, written into its `harwood.yaml`, used when a header names none |
| `images` | which site each image is from, and whether that came from its header or the default |

```example
title: Which site each frame is from
run: harwood observatory images 'obj00*.fits'
output: |
  obj001.fits
```

**A site of your own.** `set` with `--name`, `--latitude`, `--longitude`,
`--altitude` and `--timezone` adds it, in IRAF's format, to an `obsdb.dat`
beside the reduction's `harwood.yaml`, and makes it the default; every task
then finds it by its code. That file is read before IRAF's, so it can also
correct one of IRAF's sites.

```console
$ harwood observatory set skinakas --name "Skinakas Observatory" \
      --latitude 35.212 --longitude -24.8989 --altitude 1750 --timezone -2
```

Nothing is kept in your home directory: the sites, like the parameters,
travel with the reduction.
