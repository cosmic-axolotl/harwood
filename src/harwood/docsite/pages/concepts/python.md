# From Python

A task run from Python is the task run from the command line: the project
file and the preset underneath, a directory of its own under `output/` when
no output is named, an entry in the log, the timing.

```python
import harwood

result = harwood.run("daofind", image="red/obj001.fits", output="obj001.coo",
                     fwhmpsf=3.0)
```

is `harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3` -- the same
values, the same files, the same line in `harwood log`. Every example in
this documentation has a Python tab written this way.

## What comes back

| `result.` | |
|---|---|
| `outputs` | the files written |
| `data` | what was measured: `data["sources"]`, `data["shifts"]`, `data["rows"]`… |
| `warnings` | the notices, each with `.message`, `.fix` and `.level` |
| `steps` | what was done, each with what it came to |
| `duration_s` | how long it took |

## The names

The parameter names are the task's own, as `harwood help <task>` lists them
(`low_reject`, not `low-reject`). Where a task takes a list of images, one
name is a list of one, as on the command line: `input="@obj.lis"` and
`input=["obj001.fits", "obj002.fits"]` both work. A name the task does not
have is refused with the nearest ones:

```python
>>> harwood.run("daofind", image="obj001.fits", fwhm=3)
TypeError: daofind has no parameter fwhm (did you mean fwhmpsf?); `harwood help daofind` lists them
```

## Planning

`harwood.plan(...)` is `--plan`: the steps, with real values, and nothing
done.

```python
for step in harwood.plan("ccdproc", input="@obj.lis", overscan=True, trim=True):
    print(step.action, step.detail)
```

## A whole night

```python
import harwood

harwood.run("ccdproc", input="@bias.lis", overscan=True, trim=True, output="bias")
harwood.run("zerocombine", input="bias/bias00*.fits", output="Zero.fits")
harwood.run("ccdproc", input="@flat.lis", overscan=True, trim=True,
            zerocor=True, zero="Zero.fits", output="flat")
harwood.run("flatcombine", input="flat/flat00*.fits", output="Flat.fits")
harwood.run("ccdproc", input="@obj.lis", overscan=True, trim=True, zerocor=True,
            flatcor=True, zero="Zero.fits", flat="FlatV.fits", output="red")
found = harwood.run("daofind", image="red/obj001.fits", output="obj001.coo",
                    fwhmpsf=3.0)
print(found.data["sources"], "stars")
```

-- [the first night](page:start/first-night), as a script.
