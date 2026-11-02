"""artdata -- images made to order.

Images whose content is known are how the rest of the software is tested, and how a
reduction is learned without having spent a night on it. These are IRAF's: the same
patterns, with its rounding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "pattern": "A pattern of known values shows what a task does to the pixels: a section "
    "that is off, a flip, the pixel an interpolation was centred on.",
}


class MkPatternParams(TaskParams):
    input: list[str] = Field(
        ...,
        description="Images to create or to modify. A section modifies that part.",
        json_schema_extra={"positional": True},
    )
    output: list[str] = Field(
        default_factory=list, description="Images to write, when not changing the input."
    )
    pattern: Literal["constant", "grid", "checker", "coordinates", "slope", "square"] = Field(
        "constant",
        description="constant: v1. grid: lines of v2 every size pixels, on v1. "
        "checker: squares of v1 and v2. coordinates: the pixels "
        "numbered. slope: v1 to v2. square: a checker that grows.",
    )
    option: Literal["replace", "add", "multiply"] = Field(
        "replace", description="What to do with an image that exists."
    )
    v1: float = Field(0.0, description="The first value.")
    v2: float = Field(1.0, description="The second value.")
    size: int = Field(1, ge=1, description="The size of the pattern.")
    title: str = Field("", description="The title of a new image.")
    pixtype: Literal["ushort", "short", "integer", "long", "real", "double"] = Field(
        "real", description="The pixel type of a new image."
    )
    ndim: int = Field(2, ge=0, le=7, description="The dimensions of a new image.")
    ncols: int = Field(512, ge=1, description="Columns of a new image.")
    nlines: int = Field(512, ge=1, description="Lines of a new image.")
    n3: int = Field(1, ge=1, description="Pixels of its third dimension.")
    n4: int = Field(1, ge=1, description="Pixels of its fourth dimension.")
    header: str = Field(
        "", description="An image, or a file of FITS cards, for the header of a new image."
    )


def pattern_lines(pattern: str, size: int, v1: float, v2: float, nc: int, nl: int):
    """mkpatinit and mkpattern, in real: a function from IRAF's line to its pixels.
    A grid has a size of at least 2."""
    import numpy as np

    real = np.float32
    a, b = real(v1), real(v2)
    i = np.arange(nc)
    if pattern == "constant":
        first = np.full(nc, a, real)
        return lambda line: first
    if pattern == "grid":
        size = max(size, 2)
        off = np.full(nc, a, real)
        off[1::size] = b
        on = np.full(nc, b, real)
        return lambda line: on if line % size == 1 else off
    if pattern == "checker":
        odd = (i // size) % 2 == 0
        one, two = np.where(odd, a, b).astype(real), np.where(odd, b, a).astype(real)
        return lambda line: one if (max(0, line - 1) // size) % 2 == 0 else two
    if pattern == "coordinates":
        base = (i // size + 1).astype(real)
        return lambda line: (base + real(max(0, line - 1) // size * nc // size)).astype(real)
    if pattern == "slope":
        whole = (nc + nl - 2) // size
        with np.errstate(all="ignore"):
            step = real(real(b - a) / real(whole))
            base = a + ((step * i.astype(real)).astype(real) / real(size)).astype(real)
        base = base.astype(real)

        def slope(line):
            with np.errstate(all="ignore"):
                return (base + real(real(max(0, line - 1) // size) * step)).astype(real)

        return slope
    roots = np.sqrt((i // size).astype(real).astype(np.float64)).astype(real).astype(np.int64)
    even = roots % 2 == 0
    one, two = np.where(even, a, b).astype(real), np.where(even, b, a).astype(real)

    def square(line):
        at = max(0, line - 1) // size
        return one if int(np.sqrt(float(real(at)))) % 2 == 0 else two

    return square


def read_cards(name: str):
    """mkh_header: the keywords of an image, or the FITS cards of a text file,
    with what is not a FITS card passed over."""
    import re

    from astropy.io import fits

    from ..core import images

    out = fits.Header()
    if not name.strip():
        return out
    spec = images.parse(name)
    if not spec.path.exists():
        advice.fail(f"Cannot open file ({name})", "give an image, or a file of FITS cards")
    if spec.path.suffix in (".fits", ".fit", ".pl"):
        return images.read_header(spec)
    for line in spec.path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.lstrip()
        found = re.match(r"[A-Z0-9_-]{1,8} *", text)
        if found is None or found.end() < 8:
            continue
        if text[8:9] == "=" and text[9:10] != " ":
            continue
        try:
            out.append(fits.Card.fromstring(text[:80].ljust(80)))
        except (ValueError, fits.VerifyError):
            continue
    return out


PIXTYPES = {
    "ushort": "uint16",
    "short": "int16",
    "integer": "int32",
    "long": "int32",
    "real": "float32",
    "double": "float64",
}


@register
class MkPattern(Task[MkPatternParams]):
    """IRAF's mkpattern: an image, or part of one, set to a pattern, or the pattern added
    or multiplied in. A cube repeats the pattern of its first plane."""

    name = "mkpattern"
    package = "artdata"
    description = "Make or modify images with simple patterns."
    Params = MkPatternParams

    def plan(self, params: MkPatternParams) -> list[Step]:
        return [
            Step("write", f"{params.pattern} in {', '.join(params.input)}", why=WHY["pattern"])
        ]

    def execute(self, params: MkPatternParams) -> TaskResult:
        from ..core import images

        inputs = [str(s) for s in images.expand_outputs(params.input)]
        outputs = list(params.output)
        if outputs and len(outputs) != len(inputs):
            advice.fail(
                "Output image list does not match input image list",
                "give one output for each input, or none to change the input",
            )
        written = []
        for k, name in enumerate(inputs):
            written.append(self._one(name, outputs[k] if outputs else name, params))
        return TaskResult(
            outputs=written, steps=self.plan(params), data={"images": len(written)}
        )

    def _one(self, name: str, target: str, params: MkPatternParams) -> Path:
        import numpy as np
        from astropy.io import fits

        from ..core import images, imexpr, pixels, provenance

        spec = images.parse(name)
        out = images.parse(target)
        if name == target and not spec.path.exists():
            return self._new(spec.path, params)
        if not spec.path.exists():
            advice.fail(f"Cannot open image ({name})", "check the name, or give no output")
        data, header = images.read(spec)
        nl = data.shape[-2] if data.ndim > 1 else 1
        nc = data.shape[-1]
        line_of = pattern_lines(params.pattern, params.size, params.v1, params.v2, nc, nl)
        made = _lines(line_of, data.shape)
        real = np.float32
        current = data.astype(real)
        if params.option == "add":
            result = (current + made).astype(real)
        elif params.option == "multiply":
            result = (current * made).astype(real)
        else:
            result = made
        line = f"mkpattern: {params.pattern}, {params.option}"
        if name == target:
            with fits.open(str(spec.path), mode="update", memmap=False) as hdul:
                hdu = hdul[images._resolve_extension(hdul, spec.extension)]
                if spec.section:
                    box = pixels.parse_section(spec.section, hdu.data.shape)
                    hdu.data[box] = result.reshape(hdu.data[box].shape).astype(hdu.data.dtype)
                else:
                    hdu.data[...] = result.astype(hdu.data.dtype)
                provenance.stamp_header(hdu.header, [line])
            return spec.path
        kind = imexpr.pixel_type(spec)
        dtype = imexpr.STORED[kind]
        for keyword in ("BSCALE", "BZERO", "BLANK"):
            header.remove(keyword, ignore_missing=True)
        provenance.stamp_header(header, [line])
        fits.PrimaryHDU(result.astype(dtype), header).writeto(out.path, overwrite=True)
        return out.path

    def _new(self, path: Path, params: MkPatternParams) -> Path:
        import numpy as np
        from astropy.io import fits

        from ..core import provenance

        axes = [params.ncols, params.nlines, params.n3, params.n4][: params.ndim]
        shape = tuple(reversed(axes))
        # IM_LEN(out,2) is nlines whatever ndim says, and the slope goes by it.
        line_of = pattern_lines(
            params.pattern, params.size, params.v1, params.v2, axes[0], params.nlines
        )
        data = _lines(line_of, shape).astype(PIXTYPES[params.pixtype])
        header = read_cards(params.header)
        if params.title:
            header["OBJECT"] = params.title
        provenance.stamp_header(header, [f"mkpattern: {params.pattern}"])
        fits.PrimaryHDU(np.asarray(data), header).writeto(path, overwrite=True)
        return path


def _lines(line_of, shape):
    """The lines in IMIO's order. The line IRAF asks the pattern for is vout[2] - 1 after
    the read has moved on: at the last line of a plane of a cube that is 0, and of a
    one-dimensional image too; it is the only trace of it in the numbers."""
    import numpy as np

    npix = shape[-1]
    nl = shape[-2] if len(shape) > 1 else 1
    planes = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    out = np.zeros((planes, nl, npix), np.float32)
    for plane in range(planes):
        for y in range(1, nl + 1):
            last_of_plane = y == nl and len(shape) > 2
            out[plane, y - 1] = line_of(0 if len(shape) == 1 or last_of_plane else y)
    return out.reshape(shape)


# ------------------------------------------------------------------ the noise


def _cards(name: str):
    """The header of a new artdata image: IRAF's standard one by default."""
    if name == "artdata$stdheader.dat":
        name = str(Path(__file__).parent.parent / "data" / "iraf_stdheader.dat")
    return read_cards(name)


def _comment(header, text: str) -> None:
    header.add_comment(text)


def _comment_value(header, name: str, value: str) -> None:
    """mkh_comment1: %9t%s%24t%g."""
    _comment(header, f"{'':8}{name:<15}{value}")


def _start(header, task: str) -> None:
    from ..core import iraffmt

    _comment(header, f"# {iraffmt.cnvtime()}")
    _comment(header, f"begin        {task}")


def _yesno(value: bool) -> str:
    return "yes" if value else "no"


def _file_comments(header, file: Path) -> None:
    """The lines of a list that are not objects go in as they are."""
    from ..core import iraffmt

    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if len(words) < 3 or None in [iraffmt.ctod(word) for word in words[:3]]:
            _comment(header, line[:70])


class _Noise:
    """The background and noise of a line, in IRAF's order: Poisson, then read noise."""

    def __init__(self, background, gain, rdnoise, poisson, seed):
        import numpy as np

        from ..core import artdata

        self.background = np.float32(background)
        self.gain = np.float32(gain)
        self.rdnoise = np.float32(np.float32(rdnoise) / self.gain)
        self.poisson = poisson
        self.state, self.memory = artdata.new_state(seed)
        self.lib = artdata

    def line(self, data) -> None:
        """All of a line: mkpnoise of what is there with no background of its own."""
        import numpy as np

        if self.poisson:
            self.lib.mkpnoise(data, data, np.float32(0.0), self.gain, self.state, self.memory)
        if self.rdnoise > 0:
            self.lib.mkrnoise(data, self.rdnoise, self.state, self.memory)


class MkNoiseParams(TaskParams):
    input: list[str] = Field(
        ...,
        description="Images to create or to add noise to.",
        json_schema_extra={"positional": True},
    )
    output: list[str] = Field(
        default_factory=list, description="Images to write, when not changing the input."
    )
    title: str = Field("", description="The title of a new image.")
    ncols: int = Field(512, ge=1, description="Columns of a new image.")
    nlines: int = Field(512, ge=1, description="Lines of a new image.")
    header: str = Field(
        "artdata$stdheader.dat",
        description="An image, or a file of FITS cards, for a new image's header.",
    )
    background: float = Field(0.0, ge=0, description="The level to add, in data numbers.")
    gain: float = Field(1.0, gt=0, description="Electrons per data number.")
    rdnoise: float = Field(0.0, ge=0, description="The read noise, in electrons.")
    poisson: bool = Field(False, description="Add Poisson noise.")
    seed: int = Field(1, description="The seed of the random numbers.")
    cosrays: list[str] = Field(
        default_factory=list,
        description="Files of cosmic rays: x y energy. One that is not "
        "there gets the random ones written to it.",
    )
    ncosrays: int = Field(0, ge=0, description="How many random ones, if no file has them.")
    energy: float = Field(
        30000.0, ge=0, description="The highest random energy, in electrons."
    )
    radius: float = Field(0.5, ge=0, description="A cosmic ray's radius, in pixels.")
    ar: float = Field(1.0, ge=0, le=1, description="Its axial ratio.")
    pa: float = Field(0.0, description="Its position angle, in degrees.")
    comments: bool = Field(True, description="Put these parameters in the header.")


@register
class MkNoise(Task[MkNoiseParams]):
    """IRAF's mknoise: background, read and Poisson noise and cosmic rays, added to an image
    or making one. For a seed the numbers are IRAF's."""

    name = "mknoise"
    package = "artdata"
    description = "Add noise and cosmic rays to images."
    Params = MkNoiseParams

    def plan(self, params: MkNoiseParams) -> list[Step]:
        return [
            Step(
                "write",
                ", ".join(params.input),
                why="Noise of a known size is how to tell what a task can recover "
                "from what it cannot.",
            )
        ]

    def execute(self, params: MkNoiseParams) -> TaskResult:
        from ..core import images

        inputs = [str(s) for s in images.expand_outputs(params.input)]
        if params.output and len(params.output) != len(inputs):
            advice.fail(
                "Output image list does not match input image list",
                "give one output for each input, or none to change the input",
            )
        noise = _Noise(
            params.background, params.gain, params.rdnoise, params.poisson, params.seed
        )
        written = []
        for k, name in enumerate(inputs):
            target = params.output[k] if params.output else name
            cosrays = params.cosrays[k % len(params.cosrays)] if params.cosrays else ""
            written.append(self._one(name, target, cosrays, noise, params))
        return TaskResult(
            outputs=written, steps=self.plan(params), data={"images": len(written)}
        )

    def _one(self, name, target, cosrays, noise, params) -> Path:
        import numpy as np
        from astropy.io import fits

        from ..core import artdata, images, iraffmt, provenance

        real = np.float32
        spec = images.parse(name)
        new = name == target and not spec.path.exists()
        if new:
            data = np.zeros((params.nlines, params.ncols), real)
            header = _cards(params.header)
            if params.title:
                header["OBJECT"] = params.title
        else:
            if not spec.path.exists():
                advice.fail(f"Cannot open image ({name})", "check the name")
            read, header = images.read(spec)
            data = read.astype(real)
        nl, nc = data.shape
        header["GAIN"] = float(real(params.gain))
        header["RDNOISE"] = float(real(real(noise.rdnoise) * noise.gain))
        xs, ys, zs = self._rays(cosrays, nc, nl, noise, params)
        if len(xs):
            templates = artdata.Templates()
            templates.star(
                "gaussian", real(params.radius), ar=real(params.ar), pa=real(params.pa)
            )
            for j in artdata.qsort(list(range(len(xs))), ys):
                x, y, z = xs[j], ys[j], zs[j]
                obj = templates.gstar(x, y, z)
                ny, nx = obj.shape
                c1 = int(real(real(x - real(nx // 2)) + real(0.5)))
                l1 = int(real(real(y - real(ny // 2)) + real(0.5)))
                artdata.add_template(data, obj, c1, l1)
        for line in data:
            if noise.background != 0:
                line[:] = (line + noise.background).astype(real)
            noise.line(line)
        if params.comments:
            _start(header, "mknoise")
            _comment_value(header, "background", iraffmt.g(float(real(params.background))))
            _comment_value(header, "gain", iraffmt.g(float(real(params.gain))))
            _comment_value(header, "rdnoise", iraffmt.g(float(real(params.rdnoise))))
            _comment_value(header, "poisson", _yesno(params.poisson))
            _comment_value(header, "seed", str(params.seed))
            if len(xs):
                fromfile = bool(cosrays) and Path(cosrays).exists() and not params.ncosrays
                if fromfile:
                    _comment_value(header, "cosrays", cosrays)
                    _file_comments(header, Path(cosrays))
                _comment_value(header, "ncosrays", str(len(xs)))
                if not fromfile:
                    _comment_value(header, "energy", iraffmt.g(float(real(params.energy))))
                _comment_value(header, "radius", iraffmt.g(float(real(params.radius))))
                _comment_value(header, "ar", iraffmt.g(float(real(params.ar))))
                _comment_value(header, "pa", iraffmt.g(float(real(params.pa))))
        return _store(
            data, header, name, target, new, spec, "mknoise", provenance, fits, images
        )

    def _rays(self, cosrays, nc, nl, noise, params):
        """The cosmic rays: x, y and energy in data numbers."""
        import numpy as np

        from ..core import artdata, iraffmt

        real = np.float32
        xs, ys, zs = [], [], []
        if cosrays and Path(cosrays).exists():
            listed = Path(cosrays).read_text(encoding="utf-8", errors="replace")
            for line in listed.splitlines():
                words = line.split()
                numbers = [iraffmt.ctod(word) for word in words[:3]]
                if len(words) < 3 or None in numbers:
                    continue
                x, y, z = (real(number) for number in numbers)
                if x < 1 or x > nc or y < 1 or y > nl:
                    continue
                xs.append(x)
                ys.append(y)
                zs.append(real(z / noise.gain))
            return xs, ys, zs
        if params.ncosrays <= 0:
            return xs, ys, zs
        energy = real(real(params.energy) / noise.gain)
        for _ in range(params.ncosrays):
            xs.append(real(real(1.0) + real(real(nc - 1) * artdata.urand(noise.state))))
            ys.append(real(real(1.0) + real(real(nl - 1) * artdata.urand(noise.state))))
            zs.append(real(energy * artdata.urand(noise.state)))
        if cosrays:
            Path(cosrays).write_text(
                "".join(
                    f"{iraffmt.g(float(x))} {iraffmt.g(float(y))} "
                    f"{iraffmt.g(float(real(noise.gain * z)))}\n"
                    for x, y, z in zip(xs, ys, zs, strict=True)
                )
            , encoding="utf-8")
        return xs, ys, zs


def _store(data, header, name, target, new, spec, task, provenance, fits, images):
    """A new image is real; one that exists, or a copy of it, keeps its type."""
    from ..core import imexpr

    provenance.stamp_header(header, [f"{task}: {name}"])
    images.forget_limits(header)
    if new:
        path = spec.path
        fits.PrimaryHDU(data, header).writeto(path, overwrite=True)
        return path
    dtype = imexpr.STORED[imexpr.pixel_type(spec)]
    for keyword in ("BSCALE", "BZERO", "BLANK"):
        header.remove(keyword, ignore_missing=True)
    path = images.parse(target).path
    fits.PrimaryHDU(data.astype(dtype), header).writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------- mkobjects


class MkObjectsParams(TaskParams):
    input: list[str] = Field(
        ...,
        description="Images to create or to add objects to.",
        json_schema_extra={"positional": True},
    )
    objects: list[str] = Field(
        ...,
        description="A list for each image: x y magnitude, and for a galaxy its model, "
        "radius, axial ratio and position angle.",
    )
    output: list[str] = Field(
        default_factory=list, description="Images to write, when not changing the input."
    )
    title: str = Field("", description="The title of a new image.")
    ncols: int = Field(512, ge=1, description="Columns of a new image.")
    nlines: int = Field(512, ge=1, description="Lines of a new image.")
    header: str = Field(
        "artdata$stdheader.dat",
        description="An image, or a file of FITS cards, for a new image's header.",
    )
    background: float = Field(1000.0, ge=0, description="The sky of a new image, per second.")
    xoffset: float = Field(0.0, description="Added to each x.")
    yoffset: float = Field(0.0, description="Added to each y.")
    star: str = Field(
        "moffat", description="The stars' profile: moffat, gaussian, an image of a star, or "
                              "a text file with it, a value a line, out to --radius; if the "
                              "first value is 0, the file is its integral."
    )
    radius: float = Field(1.0, ge=0, description="Half the stars' full width at half maximum.")
    beta: float = Field(2.5, gt=1, description="The Moffat parameter.")
    ar: float = Field(1.0, ge=0, le=1, description="The stars' axial ratio.")
    pa: float = Field(0.0, description="Their position angle, in degrees.")
    distance: float = Field(
        1.0, gt=0, description="Divides the positions and radii, and dims."
    )
    exptime: float = Field(1.0, gt=0, description="The exposure.")
    magzero: float = Field(7.0, description="The magnitude of one data number in a second.")
    gain: float = Field(1.0, gt=0, description="Electrons per data number.")
    rdnoise: float = Field(0.0, ge=0, description="The read noise, in electrons.")
    poisson: bool = Field(False, description="Add Poisson noise.")
    seed: int = Field(1, description="The seed of the random numbers.")
    comments: bool = Field(True, description="Put these parameters in the header.")


@register
class MkObjects(Task[MkObjectsParams]):
    """IRAF's mkobjects: stars and galaxies from a list, in an image or making one. This is
    how a reduction is checked: the answer is known. For a seed the numbers are IRAF's."""

    name = "mkobjects"
    package = "artdata"
    description = "Add stars and galaxies to images, from a list."
    Params = MkObjectsParams

    def plan(self, params: MkObjectsParams) -> list[Step]:
        return [
            Step(
                "write",
                f"{', '.join(params.objects)} in {', '.join(params.input)}",
                why="Objects whose magnitudes are known are how to find out what a "
                "reduction does to them.",
            )
        ]

    def execute(self, params: MkObjectsParams) -> TaskResult:
        from ..core import images

        inputs = [str(s) for s in images.expand_outputs(params.input)]
        lists = [str(word) for item in params.objects for word in images.split_list(item)]
        if len(lists) != len(inputs):
            advice.fail(
                "Input and objects lists don't match",
                "give one list of objects for each image",
            )
        if params.output and len(params.output) != len(inputs):
            advice.fail(
                "Output image list does not match input image list",
                "give one output for each input, or none to change the input",
            )
        from ..core import artdata

        noise = _Noise(0.0, params.gain, params.rdnoise, params.poisson, params.seed)
        templates = artdata.Templates()
        written, warnings = [], []
        for k, name in enumerate(inputs):
            target = params.output[k] if params.output else name
            written.append(
                self._one(name, target, lists[k], noise, templates, warnings, params)
            )
        return TaskResult(
            outputs=written,
            steps=self.plan(params),
            warnings=warnings,
            data={"images": len(written)},
        )

    def _one(self, name, target, listing, noise, templates, warnings, params) -> Path:
        import numpy as np
        from astropy.io import fits

        from ..core import artdata, images, iraffmt, provenance

        real = np.float32
        file = Path(listing)
        if not file.is_file():
            advice.fail(f"Cannot open file ({listing})", "check the name of the list")
        spec = images.parse(name)
        new = name == target and not spec.path.exists()
        background = real(real(params.exptime) * real(params.background))
        if new:
            data = np.zeros((params.nlines, params.ncols), real)
            header = _cards(params.header)
            if params.title:
                header["OBJECT"] = params.title
            header["EXPTIME"] = float(real(params.exptime))
            header["GAIN"] = float(real(params.gain))
            header["RDNOISE"] = float(real(real(noise.rdnoise) * noise.gain))
        else:
            if not spec.path.exists():
                advice.fail(f"Cannot open image ({name})", "check the name")
            read, header = images.read(spec)
            data = read.astype(real)
        nl, nc = data.shape
        templates.star(
            params.star,
            real(params.radius),
            real(params.beta),
            real(params.ar),
            real(params.pa),
        )
        objects, ys = self._read(file, nc, nl, templates, warnings, params)

        def with_noise(row, target_row):
            if params.poisson:
                artdata.mkpnoise(
                    row, target_row, background, noise.gain, noise.state, noise.memory
                )
            if noise.rdnoise > 0:
                artdata.mkrnoise(target_row, noise.rdnoise, noise.state, noise.memory)

        for j in artdata.qsort(list(range(len(objects))), ys):
            kind, x, y, z, r, ar, pa = objects[j]
            if kind is None:
                obj = templates.gstar(x, y, z)
            else:
                obj = templates.gobject(kind, x, y, z, r, ar, pa)
                if obj is None:
                    continue
            ny, nx = obj.shape
            artdata.add_template(
                data,
                obj,
                artdata.nint(x) - nx // 2,
                artdata.nint(y) - ny // 2,
                None if new else with_noise,
            )
        if new:
            for line in data:
                if background != 0:
                    line[:] = (line + background).astype(real)
                noise.line(line)
        if params.comments:
            _start(header, "mkobjects")
            _comment_value(header, "objects", listing)
            for key in ("background", "xoffset", "yoffset"):
                _comment_value(header, key, iraffmt.g(float(real(getattr(params, key)))))
            _comment_value(header, "star", params.star)
            for key in (
                "radius",
                "beta",
                "ar",
                "pa",
                "distance",
                "exptime",
                "magzero",
                "gain",
                "rdnoise",
            ):
                _comment_value(header, key, iraffmt.g(float(real(getattr(params, key)))))
            _comment_value(header, "poisson", _yesno(params.poisson))
            _comment_value(header, "seed", str(params.seed))
            _file_comments(header, file)
        return _store(
            data, header, name, target, new, spec, "mkobjects", provenance, fits, images
        )

    def _read(self, file, nc, nl, templates, warnings, params):
        """The list: the stars and known galaxies that land on the image."""
        import numpy as np

        from ..core import iraffmt

        real = np.float32
        distance = real(params.distance)
        objects, ys = [], []
        for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
            words = line.split()
            first = [iraffmt.ctod(word) for word in words[:3]]
            if len(words) < 3 or None in first:
                continue
            x, y, z = (real(number) for number in first)
            x = real(real(params.xoffset) + real(x / distance))
            y = real(real(params.yoffset) + real(y / distance))
            if x < 1 or x > nc or y < 1 or y > nl:
                continue
            exponent = float(real(real(-0.4) * real(z - real(params.magzero))))
            flux = real(
                float(real(real(params.exptime) / real(distance * distance)))
                * pow(10.0, exponent)
            )
            kind, r, ar, pa = None, real(0.0), real(0.0), real(0.0)
            numbers = [iraffmt.ctod(word) for word in words[4:7]]
            if len(words) >= 7 and None not in numbers:
                kind = words[3]
                r = real(real(numbers[0]) / distance)
                ar = real(numbers[1])
                pa = real(real(numbers[2]) / real(57.295779513082320877))
                if templates.objects.get(kind, False) is False:
                    from ..core.artdata import object_profile

                    templates.objects[kind] = object_profile(kind, templates.dynrange)
                    if templates.objects[kind] is None:
                        warnings.append(
                            advice.warn(
                                f"Object template {kind} not found",
                                "use sersic, expdisk, devauc, an image or a text file",
                            )
                        )
            objects.append((kind, x, y, flux, r, ar, pa))
            ys.append(y)
        return objects, ys


# ----------------------------------------------------------------- starlist


class StarListParams(TaskParams):
    starlist: str = Field(
        ...,
        description="The file to write; a list is added to its end.",
        json_schema_extra={"positional": True},
    )
    nstars: int = Field(
        5000, ge=1, description="How many stars.", json_schema_extra={"positional": True}
    )
    spatial: Literal["uniform", "hubble", "file"] = Field(
        "uniform", description="Where they are: file takes a density against radius from "
                               "--sfile."
    )
    sfile: str = Field("", description="For spatial file: radius and relative density, a "
                                       "line.")
    xmin: float = Field(1.0, description="The lowest x.")
    xmax: float = Field(512.0, description="The highest x.")
    ymin: float = Field(1.0, description="The lowest y.")
    ymax: float = Field(512.0, description="The highest y.")
    xcenter: float | None = Field(None, description="The centre of a hubble distribution.")
    ycenter: float | None = Field(None, description="The centre of a hubble distribution.")
    core_radius: float = Field(30.0, description="Its core radius.")
    base: float = Field(0.0, description="Its relative background density.")
    sseed: int = Field(1, description="The seed for the positions.")
    luminosity: Literal["uniform", "salpeter", "bands", "powlaw", "file"] = Field(
        "powlaw", description="How bright they are: file takes a function from --lfile."
    )
    lfile: str = Field("", description="For luminosity file: magnitude and relative "
                                       "probability, a line.")
    minmag: float = Field(-7.0, description="The lowest magnitude.")
    maxmag: float = Field(0.0, description="The highest magnitude.")
    mzero: float = Field(
        -4.0, description="The absolute magnitude zero point for salpeter and bands."
    )
    power: float = Field(0.6, description="The exponent for powlaw.")
    alpha: float = Field(0.74, description="Alpha for bands.")
    beta: float = Field(0.04, description="Beta for bands.")
    delta: float = Field(0.294, description="Delta for bands.")
    mstar: float = Field(1.28, description="Mstar for bands.")
    lseed: int = Field(1, description="The seed for the magnitudes.")
    nssample: int = Field(100, description="Points of the spatial function.")
    sorder: int = Field(10, description="Spline pieces for its inverse.")
    nlsample: int = Field(100, description="Points of the luminosity function.")
    lorder: int = Field(10, description="Spline pieces for its inverse.")


def _amap(values, a1, a2, b1, b2):
    """amapr: the map, in real, kept within b1 and b2."""
    import numpy as np

    real = np.float32
    scale = real(real(real(b2) - real(b1)) / real(real(a2) - real(a1)))
    mapped = ((np.asarray(values, real) - real(a1)).astype(real) * scale).astype(real)
    low, high = min(real(b1), real(b2)), max(real(b1), real(b2))
    return np.minimum(high, np.maximum(low, (mapped + real(b1)).astype(real)))


def _inverse(fractions, values, order):
    """cvfit of what the sample comes back through: value against the cumulative."""
    import numpy as np

    from ..core import curfit

    real = np.float32
    curve = curfit.Curve(curfit.SPLINE3, order, real(0.0), real(1.0))
    curve.accumulate(
        np.asarray(fractions, real), np.asarray(values, real), np.ones(len(values), real)
    )
    status = curve.solve()
    return curve if status == curfit.OK else None


def _hubble(params, state):
    import math

    import numpy as np

    from ..core import artdata

    real = np.float32
    xmin, xmax, ymin, ymax = (
        real(v) for v in (params.xmin, params.xmax, params.ymin, params.ymax)
    )
    half = real(2.0)
    xc = real(params.xcenter) if params.xcenter is not None else real(real(xmax + xmin) / half)
    yc = real(params.ycenter) if params.ycenter is not None else real(real(ymax + ymin) / half)

    def square(v):
        return real(v * v)

    def dist(a, b):
        return real(square(real(a - xc)) + square(real(b - yc)))

    corners = (dist(xmin, ymin), dist(xmax, ymin), dist(xmax, ymax), dist(xmin, ymax))
    if xmin <= xc <= xmax and ymin <= yc <= ymax:
        rmin = real(0.0)
    elif ymin <= yc <= ymax:
        rmin = min(abs(real(xmin - xc)), abs(real(xmax - xc)))
    elif xmin <= xc <= xmax:
        rmin = min(abs(real(ymin - yc)), abs(real(ymax - yc)))
    else:
        rmin = real(np.sqrt(min(corners)))
    rmax = real(np.sqrt(max(corners)))
    n = params.nssample
    dr = real(real(rmax - rmin) / real(n - 1))
    core = real(params.core_radius)
    base = real(params.base)
    r1 = square(core)
    rad, prob = np.zeros(n, real), np.zeros(n, real)
    rval = rmin
    for i in range(n):
        rad[i] = rval
        r2 = real(real(core + rval) / core)
        first = float(r1) * (math.log(float(r2)) + float(real(real(1.0) / r2)) - 1.0)
        prob[i] = real(first + float(real(real(base * square(rval)) / half)))
        rval = real(rval + dr)
    prob = _amap(prob, prob.min(), prob.max(), 0.0, 1.0)
    curve = _inverse(prob, rad, params.sorder)
    if curve is None:
        n = params.nstars
        return np.full(n, real(real(xmin + xmax) / half)), np.full(
            n, real(real(ymin + ymax) / half)
        )
    xs, ys = [], []
    while len(xs) < params.nstars:
        rval = real(curve(artdata.urand(state)))
        theta = real(real(real(360.0) * artdata.urand(state)) / artdata.RADIAN)
        x = real(float(rval) * math.cos(float(theta)) + float(xc))
        y = real(float(rval) * math.sin(float(theta)) + float(yc))
        if xmin <= x <= xmax and ymin <= y <= ymax:
            xs.append(x)
            ys.append(y)
    return np.array(xs, real), np.array(ys, real)


def _xy_file(name: str, what: str):
    """st_gfetchxy: the first two numbers of each line that has them."""
    import numpy as np

    from ..core import iraffmt

    path = Path(name)
    if not path.is_file():
        advice.fail(f"Cannot open file ({name})", f"give the {what} with its option",
                    FileNotFoundError)
    xs, ys = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if len(words) < 2:
            continue
        a, b = iraffmt.ctod(words[0]), iraffmt.ctod(words[1])
        if a is None or b is None:
            continue
        xs.append(a)
        ys.append(b)
    return np.array(xs, np.float32), np.array(ys, np.float32)


def _smoothed(x, y, order):
    """cvfit of the relative probability, its pieces held back to a quarter of the points."""
    import numpy as np

    from ..core import curfit

    real = np.float32
    curve = curfit.Curve(curfit.SPLINE3, min(order, max(1, x.size // 4)), x.min(), x.max())
    curve.accumulate(x, y, np.ones(x.size, real))
    return curve if curve.solve() == curfit.OK else None


def _integrated(values, order, weights):
    """asigrl across the sample, mapped to 0..1, and the inverse fit to it."""
    import numpy as np

    from ..core import asi

    real = np.float32
    interpolant = asi.Interpolant("spline3", np.asarray(weights, real))
    integral = np.zeros(len(values), real)
    for i in range(1, len(values)):
        piece = real(interpolant.integral(float(i), float(i + 1)))
        integral[i] = real(integral[i - 1] + piece)
    integral = _amap(integral, integral.min(), integral.max(), 0.0, 1.0)
    return _inverse(integral, values, order)


def _spatial_file(params, state):
    """st_sfsample."""
    import math

    import numpy as np

    from ..core import artdata

    real = np.float32
    half = real(2.0)
    xmin, xmax, ymin, ymax = (real(v) for v in (params.xmin, params.xmax, params.ymin,
                                                params.ymax))
    middle = (np.full(params.nstars, real(real(xmin + xmax) / half)),
              np.full(params.nstars, real(real(ymin + ymax) / half)))
    r, rprob = _xy_file(params.sfile, "spatial function")
    if r.size == 0:
        return middle
    xc = real(params.xcenter) if params.xcenter is not None else real(real(xmax + xmin) / half)
    yc = real(params.ycenter) if params.ycenter is not None else real(real(ymax + ymin) / half)
    smooth = _smoothed(r, rprob, params.sorder)
    if smooth is None:
        return middle
    n = params.nssample
    rval = r.min()
    dr = real(real(r.max() - r.min()) / real(n - 1))
    rad, prob = np.zeros(n, real), np.zeros(n, real)
    for i in range(n):
        rad[i] = rval
        prob[i] = real(rval * real(smooth(rval)))
        rval = real(rval + dr)
    curve = _integrated(rad, params.sorder, prob)
    if curve is None:
        return middle
    xs, ys = [], []
    while len(xs) < params.nstars:
        rval = real(curve(artdata.urand(state)))
        theta = real(real(real(360.0) * artdata.urand(state)) / artdata.RADIAN)
        x = real(float(rval) * math.cos(float(theta)) + float(xc))
        y = real(float(rval) * math.sin(float(theta)) + float(yc))
        if xmin <= x <= xmax and ymin <= y <= ymax:
            xs.append(x)
            ys.append(y)
    return np.array(xs, real), np.array(ys, real)


def _luminosity_file(params, state):
    """st_lfsample."""
    import numpy as np

    from ..core import artdata, curfit

    real = np.float32
    minmag, maxmag = real(params.minmag), real(params.maxmag)
    middle = np.full(params.nstars, real(real(minmag + maxmag) / real(2.0)))
    m, mprob = _xy_file(params.lfile, "luminosity function")
    if m.size == 0:
        return middle
    smooth = curfit.Curve(curfit.SPLINE3, max(1, m.size // 4), m.min(), m.max())
    smooth.accumulate(m, mprob, np.ones(m.size, real))
    if smooth.solve() != curfit.OK:
        return middle
    n = params.nlsample
    mval = max(minmag, m.min())
    dm = real(real(min(maxmag, m.max()) - mval) / real(n - 1))
    mags, prob = np.zeros(n, real), np.zeros(n, real)
    for i in range(n):
        mags[i] = mval
        prob[i] = real(smooth(mval))
        mval = real(mval + dm)
    curve = _integrated(mags, params.lorder, prob)
    if curve is None:
        return middle
    return np.array([real(curve(artdata.urand(state))) for _ in range(params.nstars)], real)


def _sampled(values, probabilities, params, state):
    """The integral of a luminosity function and the magnitudes back through its inverse."""
    import numpy as np

    from ..core import artdata, asi

    real = np.float32
    interpolant = asi.Interpolant("spline3", np.asarray(probabilities, real))
    integral = np.zeros(len(values), real)
    for i in range(1, len(values)):
        integral[i] = real(
            integral[i - 1] + real(interpolant.integral(float(i), float(i + 1)))
        )
    integral = _amap(integral, integral.min(), integral.max(), 0.0, 1.0)
    curve = _inverse(integral, values, params.lorder)
    if curve is None:
        middle = real(real(real(params.minmag) + real(params.maxmag)) / real(2.0))
        return np.full(params.nstars, middle)
    zero = real(params.mzero)
    return np.array(
        [real(real(curve(artdata.urand(state))) + zero) for _ in range(params.nstars)], real
    )


def _luminosities(params, state):
    import math

    import numpy as np

    from ..core import artdata

    real = np.float32
    nstars, n = params.nstars, params.nlsample
    minmag, maxmag = real(params.minmag), real(params.maxmag)
    if params.luminosity == "file":
        return _luminosity_file(params, state)
    if params.luminosity == "uniform":
        values = [artdata.urand(state) for _ in range(nstars)]
        return _amap(values, 0.0, 1.0, minmag, maxmag)
    if params.luminosity == "powlaw":
        power = real(params.power)
        a = real(math.pow(10.0, float(real(power * real(maxmag - minmag)))) - 1.0)
        out = []
        for _ in range(nstars):
            inside = real(real(a * artdata.urand(state)) + real(1.0))
            out.append(
                real(
                    minmag
                    + real(0.43429448190325182765 * math.log(float(inside)) / float(power))
                )
            )
        return np.array(out, real)
    zero = real(params.mzero)
    if params.luminosity == "salpeter":
        low, high = real(-4.0), real(16.0)
    else:
        low, high = real(-6.0), real(19.0)
    magval = max(real(minmag - zero), low)
    dmag = real(real(min(real(maxmag - zero), high) - magval) / real(n - 1))
    mags, prob = np.zeros(n, real), np.zeros(n, real)
    alpha, beta, delta, mstar = (
        real(v) for v in (params.alpha, params.beta, params.delta, params.mstar)
    )
    for i in range(n):
        mags[i] = magval
        if params.luminosity == "salpeter":
            exponent = real(
                real(real(-3.158375) + real(real(1.550629e-1) * magval))
                - real(real(5.19388e-3) * real(magval * magval))
            )
            prob[i] = real(math.pow(10.0, float(exponent)))
        else:
            mtemp = real(15.0) if magval > 15.0 else real(magval - mstar)
            temp1 = real(math.pow(10.0, float(real(beta * mtemp))))
            inner = 1.0 + math.pow(10.0, float(real(real(real(beta - alpha) * delta) * mtemp)))
            temp2 = real(math.pow(inner, float(real(real(1.0) / delta))))
            prob[i] = real(temp1 / temp2)
        magval = real(magval + dmag)
    return _sampled(mags, prob, params, state)


@register
class StarList(Task[StarListParams]):
    """IRAF's starlist: a list of positions and magnitudes for mkobjects, drawn from a
    spatial and a luminosity function. For a seed the numbers are IRAF's."""

    name = "starlist"
    package = "artdata"
    description = "Make a list of artificial stars."
    Params = StarListParams

    def plan(self, params: StarListParams) -> list[Step]:
        return [
            Step(
                "write",
                f"{params.nstars} stars to {params.starlist}",
                why="A list for mkobjects, whose answer is known.",
            )
        ]

    def execute(self, params: StarListParams) -> TaskResult:
        import numpy as np

        from ..core import artdata, iraffmt

        real = np.float32
        sstate, _ = artdata.new_state(params.sseed)
        lstate, _ = artdata.new_state(params.lseed)
        if params.spatial == "uniform":
            drawn = [artdata.urand(sstate) for _ in range(2 * params.nstars)]
            x = _amap(drawn[0::2], 0.0, 1.0, params.xmin, params.xmax)
            y = _amap(drawn[1::2], 0.0, 1.0, params.ymin, params.ymax)
        elif params.spatial == "file":
            x, y = _spatial_file(params, sstate)
        else:
            x, y = _hubble(params, sstate)
        mag = _luminosities(params, lstate)

        def g(value):
            return iraffmt.g(float(real(value)))

        half = real(2.0)
        lines = [
            f"# {iraffmt.cnvtime()}",
            f"# begin\t{params.starlist}",
            f"#\tspatial\t\t{params.sfile if params.spatial == 'file' else params.spatial}",
        ]
        if params.spatial in ("hubble", "file"):
            xc = (
                real(params.xcenter)
                if params.xcenter is not None
                else real(real(real(params.xmax) + real(params.xmin)) / half)
            )
            yc = (
                real(params.ycenter)
                if params.ycenter is not None
                else real(real(real(params.ymax) + real(params.ymin)) / half)
            )
            lines += [f"#\txcenter\t\t{g(xc)}", f"#\tycenter\t\t{g(yc)}"]
            if params.spatial == "hubble":
                lines += [
                    f"#\tcoreradius\t{g(params.core_radius)}",
                    f"#\tbaseline\t{g(params.base)}",
                ]
        lines += [
            f"#\txmin\t\t{g(params.xmin)}",
            f"#\txmax\t\t{g(params.xmax)}",
            f"#\tymin\t\t{g(params.ymin)}",
            f"#\tymax\t\t{g(params.ymax)}",
            "#\tluminosity\t"
            + (params.lfile if params.luminosity == "file" else params.luminosity),
        ]
        if params.luminosity == "powlaw":
            lines.append(f"#\tpower\t\t{g(params.power)}")
        elif params.luminosity == "salpeter":
            lines.append(f"#\tmzero\t\t{g(params.mzero)}")
        elif params.luminosity == "bands":
            lines += [
                f"#\tmzero\t\t{g(params.mzero)}",
                f"#\talpha\t\t{g(params.alpha)}",
                f"#\tbeta\t\t{g(params.beta)}",
                f"#\tdelta\t\t{g(params.delta)}",
                f"#\tmstar\t\t{g(params.mstar)}",
            ]
        lines += [
            f"#\tminmag\t\t{g(params.minmag)}",
            f"#\tmaxmag\t\t{g(params.maxmag)}",
            f"#\tnssample\t{params.nssample}",
            f"#\tsorder\t\t{params.sorder}",
            f"#\tsseed\t\t{params.sseed}",
            f"#\tnlsample\t{params.nlsample}",
            f"#\tlorder\t\t{params.lorder}",
            f"#\tlseed\t\t{params.lseed}",
            f"#\tnstars\t\t{params.nstars}",
        ]
        lines += _star_lines(x, y, mag)
        target = Path(params.starlist)
        with target.open("a", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")
        return TaskResult(
            outputs=[target], steps=self.plan(params), data={"stars": params.nstars}
        )


def _star_lines(x, y, mag) -> list[str]:
    """st_dtwrite: in order of y."""
    import numpy as np

    from ..core import iraffmt

    first, second = iraffmt.Format("%8.3f"), iraffmt.Format("%7.3f")
    order = np.argsort(y, kind="stable")
    return [
        f"\t{first(float(x[j]))}  {first(float(y[j]))}  {second(float(mag[j]))}" for j in order
    ]
