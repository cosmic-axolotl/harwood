Sorts a night into lists, as IRAF's ccdgroups does: one file per group,
each holding the names of its images, ready to be given as an `@list`.

| `--group` | a group shares |
|---|---|
| `ccdtype` | the image type: `<root>zero`, `<root>flat`, `<root>object` |
| `subset` | the filter |
| `title` | the `OBJECT` |
| `date` | the date of observation, without the time |
| `position` | the pointing, within `--radius` arcseconds of the group's first image |

```example
title: The night, by image type
run: harwood ccdgroups 'bias00*.fits' 'flat00*.fits' 'obj00*.fits' night --group ccdtype
output: |
  -> nightobject
```

`nightobject` now lists the three object frames, and `harwood ccdproc
@nightobject ...` processes them. Groups by number (`night1`, `night2`, …)
are numbered in the order their first image is met.

```warning
As in IRAF, the group files are appended to: run it twice and every name is
listed twice. harwood says so when it finds the files already there; delete
them to start over.
```
