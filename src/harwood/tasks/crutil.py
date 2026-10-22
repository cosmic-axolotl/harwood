"""crutil -- cosmic rays.

IRAF's `cosmicrays` is from 1989 and works on a flux ratio: a pixel standing
above its neighbours, with the neighbours dark. It cannot separate a cosmic ray
from the core of an undersampled star, because on that statistic the two are
the same thing -- which is why the task is interactive, and why its own help
page tells you to check the result by eye.

The default here is L.A.Cosmic (van Dokkum 2001), which separates them by
asking a different question: is this edge sharper than the optics can make?
IRAF's algorithm is still available, because reproducing an old reduction
needs the method it was done with.

What is new is neither algorithm. It is `--verify`: given several exposures of
the same field, a cosmic ray falls on one and not the others, which is ground
truth. The detector can then be *scored on your own data* -- completeness and
false positives, measured -- instead of trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult, format_field, uncertainty_field

WHY = {
    "read": "The frame, and the gain and read noise that turn its counts into "
            "a noise estimate. Those two numbers are not cosmetic here: the "
            "whole method is a comparison against the expected noise, so a "
            "gain that is wrong by a factor of two moves every threshold.",
    "slit": "On a long-slit frame the object is a line, not a point, and the "
            "usual fine-structure model -- built to recognise a star -- sees "
            "nothing on the smooth flank of a spectrum. A second model, a "
            "median along the dispersion, estimates the spectrum without the "
            "cosmic rays in it, because a track is local along the dispersion "
            "and a median steps over it. Whichever model finds more real "
            "structure wins -- but only where the pixel agrees with the "
            "model, or a cosmic ray sitting on the spectrum would be "
            "protected along with it. Measured over six synthetic long-slit "
            "frames at the 1992 detector's gain: false detections on the "
            "flanks 114 without it and 0 with, for three tenths of a per cent "
            "of completeness.",
    "detect": "A cosmic ray deposits its energy in one pass, so it makes a "
              "mark sharper than the telescope can focus. L.A.Cosmic finds "
              "that by convolving with a Laplacian at twice the sampling -- "
              "the subsampling is what makes it respond to the edge rather "
              "than to the pixel grid -- and dividing by the local noise. A "
              "second image, the fine structure, models how much unresolved "
              "structure a real object would have there: a star is sharp but "
              "sits on a wing, a cosmic ray is sharp and alone.",
    "grow": "A track is bright in the middle and marginal at its ends, so "
            "pixels touching a detection are judged against a lower "
            "threshold. Whether the fine-structure test still applies while "
            "growing is the difference between a method that occasionally "
            "eats a star and one that does not.",
    "clean": "Each flagged pixel is replaced by the median of its *unflagged* "
             "neighbours. Unflagged matters: a track is several pixels, and a "
             "median that includes its neighbours swaps one bad value for "
             "another. Nothing is interpolated across a gap wider than the "
             "window, and the mask records what was touched.",
    "verify": "Several exposures of one field disagree only where something "
              "happened in one of them. That is ground truth for cosmic rays, "
              "and it lets the single-frame detector be measured rather than "
              "believed -- on these frames, with this instrument, at this "
              "threshold, rather than from the numbers in a paper. The frames "
              "are scaled to a common level first, because transparency "
              "changes between exposures and an unscaled comparison counts "
              "that real change as a cosmic ray in every bright pixel.",
    "write": "The cleaned frame, and the mask beside it. The mask is the part "
             "worth keeping: it says which pixels are no longer measurements, "
             "so anything downstream can decline to use them rather than "
             "quietly averaging an interpolation.",
}


class CosmicRaysParams(TaskParams):
    input: list[str] = Field(
        ..., description="Frames to clean. Globs and @lists expand.",
        json_schema_extra={"positional": True},
    )
    output: str = Field(
        "",
        description="Where to write. A directory, or a prefix. Left out, it "
                    "goes to this run's own directory.",
    )
    crmask: str = Field(
        "",
        description="Also write the cosmic rays as a mask image of their own, "
                    "1 where a pixel was hit: a directory or a prefix. Left out, "
                    "none -- they are in the DQ of the frame, bit 8.",
    )
    algorithm: str = Field(
        "lacosmic",
        description="lacosmic (van Dokkum 2001) or iraf (the 1989 flux ratio).",
    )
    gain: float | str | None = Field(
        None, description="Electrons per count, or the header keyword holding it. "
                          "Left out, the frames' GAIN, else 1 -- and the run says so.")
    rdnoise: float | str | None = Field(
        None, description="Read noise in electrons, or the keyword holding it. Left "
                          "out, the frames' RDNOISE, else 0.")
    sigclip: float = Field(
        4.5, gt=0,
        description="How many sigma above the noise a Laplacian edge must be.",
    )
    sigfrac: float = Field(
        0.3, gt=0, le=1.0,
        description="Threshold for pixels touching a detection, as a fraction "
                    "of sigclip.",
    )
    objlim: float = Field(
        5.0, gt=0,
        description="Ratio of edge to fine structure. Lower it first if real "
                    "objects are being eaten.",
    )
    niter: int = Field(4, ge=1, description="Detection passes.")
    satlevel: float = Field(
        65535.0, gt=0, description="Counts at which a pixel is saturated and left alone."
    )
    strict_growth: bool = Field(
        True,
        description="Keep the fine-structure test while growing into "
                    "neighbours. Off is the published behaviour, and is what "
                    "to use when reproducing someone else's reduction.",
    )
    dispaxis: int | None = Field(
        None, ge=1, le=2,
        description="1 or 2 for a long-slit spectrum, so the fine-structure "
                    "model knows the object is a line rather than a point. "
                    "Read from the DISPAXIS keyword when left out.",
    )
    fast: bool = Field(
        False,
        description="Approximate the medians as row-then-column, about three "
                    "times faster on a large frame.",
    )
    threshold: float = Field(
        25.0, description="IRAF algorithm: counts above the background."
    )
    fluxratio: float = Field(
        2.0, description="IRAF algorithm: neighbour-to-peak ratio, per cent."
    )
    npasses: int = Field(5, ge=1, description="IRAF algorithm: passes.")
    verify: bool = Field(
        False,
        description="Score the detector against what the frames agree on. "
                    "Needs at least three exposures of the same field, "
                    "already aligned.",
    )
    clean: bool = Field(
        True, description="Replace the hit pixels by the median of their unflagged "
                          "neighbours, as IRAF does. Off, they are only flagged in "
                          "the DQ and the data is left as it was -- what imcombine, "
                          "which leaves flagged pixels out, wants.")
    uncertainty: Literal["VAR", "ERR", "UNCERT"] = uncertainty_field()
    format: Literal["harwood", "iraf"] = format_field()


@register
class CosmicRays(Task[CosmicRaysParams]):
    name = "cosmicrays"
    package = "crutil"
    description = "Find and remove cosmic rays, and measure how well it worked."
    Params = CosmicRaysParams

    def plan(self, params: CosmicRaysParams) -> list[Step]:
        steps = [
            Step("read", ", ".join(params.input), why=WHY["read"]),
            Step("detect", f"{params.algorithm}, {params.sigclip} sigma"
                 + (f", dispersion along axis {params.dispaxis}"
                    if params.dispaxis else ""), why=WHY["detect"]),
        ]
        if params.algorithm == "lacosmic":
            steps.append(Step(
                "grow", f"{params.sigfrac} x sigclip"
                        + (", fine structure still applies" if params.strict_growth
                           else ", as published"), why=WHY["grow"]))
        steps.append(Step("clean", "median of unflagged neighbours",
                          why=WHY["clean"], skipped=not params.clean,
                          reason="" if params.clean else "--no-clean"))
        if params.verify:
            steps.append(Step("verify", "against the stack", why=WHY["verify"]))
        steps.append(Step("write", params.output or "this run's directory",
                          why=WHY["write"]))
        return steps

    def execute(self, params: CosmicRaysParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import cosmic, images, output

        if params.algorithm not in ("lacosmic", "iraf"):
            advice.fail(
                f"no algorithm called '{params.algorithm}'.",
                "use 'lacosmic' (van Dokkum 2001, the default) or 'iraf' (the "
                "1989 flux ratio, for reproducing an old reduction).")

        frames = images.expand(params.input)
        if not frames:
            advice.fail(f"nothing matched {', '.join(params.input)}.",
                        "check the path and the pattern; 'harwood sections' "
                        "shows what a template expands to.")

        params, said = _noise(params, images.read_header(frames[0]))
        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        detector = self._detector(params, cosmic)

        warnings, rows, written = list(said), [], []
        loaded, results = [], []
        for spec in frames:
            data, header = images.read(spec)
            if data.ndim != 2:
                advice.fail(
                    f"'{spec.path}' is {data.ndim}-dimensional.",
                    "cosmic-ray detection works on a 2-D frame; give a "
                    "section if the file holds a cube.")
            axis = params.dispaxis or _header_dispaxis(header)
            found = (detector(data, axis) if params.algorithm == "lacosmic"
                     else detector(data))
            loaded.append((spec, data, header))
            results.append(found)
            rows.append({
                "image": Path(spec.path).name,
                "found": found.count,
                "per cent": round(100.0 * found.fraction, 4),
                "passes": found.passes,
                "per pass": " ".join(str(n) for n in found.found_per_pass),
            })

        total = sum(item.count for item in results)
        pixels = sum(item.mask.size for item in results)
        by_action["read"].measured = {
            "frames": len(frames),
            "gain": f"{params.gain} e-/count",
            "read noise": f"{params.rdnoise} e-",
        }
        by_action["detect"].measured = {
            "flagged": total,
            "of the frame": f"{100.0 * total / max(pixels, 1):.3f}%",
            "most in one frame": max(item.count for item in results),
        }
        if params.algorithm == "lacosmic":
            by_action["grow"].measured = {
                "first pass": sum(item.found_per_pass[0] for item in results),
                "later passes": total - sum(item.found_per_pass[0]
                                            for item in results),
            }

        # A frame that loses an implausible share of itself is a frame whose
        # gain or read noise is wrong, not a frame full of cosmic rays.
        worst = max(results, key=lambda item: item.fraction)
        if worst.fraction > 0.01:
            warnings.append(advice.problem(
                f"{100 * worst.fraction:.2f} per cent of a frame was flagged. "
                "A dark exposure on a quiet site collects of order 0.01 per "
                "cent; a hundred times that is the detector's noise model "
                "being wrong, not the sky.",
                f"check --gain (now {params.gain}) and --rdnoise (now "
                f"{params.rdnoise}) against the header. Too low a gain makes "
                "every pixel look significant."))

        if params.verify:
            self._verify(params, loaded, detector, by_action, warnings, cosmic)

        for (spec, data, header), found in zip(loaded, results, strict=True):
            files, notes = self._write(params, spec, data, header, found, output, fits, np)
            written += files
            if notes and not any("--format iraf" in str(w) for w in warnings):
                warnings.extend(notes)

        by_action["write"].measured = {"files": len(written)}
        return TaskResult(outputs=written, steps=steps, warnings=warnings,
                          data={"rows": rows},
                          history=[f"cosmicrays: {params.algorithm}, "
                                   f"{total} pixels flagged"])

    def _detector(self, params: CosmicRaysParams, cosmic):
        """One frame in, a Detection out -- the thing `verify` also scores."""
        if params.algorithm == "iraf":
            def run(data):
                return cosmic.iraf_cosmicrays(
                    data, threshold=params.threshold,
                    fluxratio=params.fluxratio, npasses=params.npasses)
            return run

        def run(data, dispaxis=None):
            return cosmic.lacosmic(
                data, gain=params.gain, rdnoise=params.rdnoise,
                sigclip=params.sigclip, sigfrac=params.sigfrac,
                objlim=params.objlim, niter=params.niter,
                satlevel=params.satlevel, strict_growth=params.strict_growth,
                separable=params.fast,
                dispaxis=params.dispaxis if dispaxis is None else dispaxis)
        return run

    def _verify(self, params, loaded, detector, by_action, warnings, cosmic):
        """Score the detector against what the frames agree on."""
        shapes = {item[1].shape for item in loaded}
        if len(shapes) > 1:
            advice.fail(
                "the frames are not the same shape, so they cannot be "
                f"compared pixel by pixel: {', '.join(str(s) for s in shapes)}.",
                "trim them to a common section, or run imalign first.")

        score = cosmic.verify([item[1] for item in loaded], detector,
                              gain=params.gain, rdnoise=params.rdnoise)
        by_action["verify"].measured = {
            "the frames agree on": sum(row["frames agree"] for row in score["rows"]),
            "scaled by": " ".join(f"{value:.3f}" for value in score["scaled_by"]),
            "completeness": f"{100 * score['completeness']:.1f}%",
            "not supported by the stack": f"{100 * score['false_positive']:.1f}%",
        }
        # A score computed on frames that are not the same scene measures the
        # scene, not the detector. It has to be said before the number is.
        if not score["static"]:
            warnings.append(advice.problem(
                "the frames disagree far more on bright pixels than on faint "
                f"ones ({100 * score['rate_bright']:.2f} per cent against "
                f"{100 * score['rate_faint']:.2f}). A cosmic ray does not care "
                "how bright the pixel it lands on is, so this is the scene "
                "changing between exposures -- a source that varied, a profile "
                "that moved with the seeing, a trace that drifted. The "
                "completeness below is measuring that, not the detector.",
                "verify against frames of a static field -- darks, or a set "
                "where nothing moved. For these, judge the detections by "
                "looking at them, and reach for imcombine with rejection, "
                "which uses the same information without needing the scene to "
                "hold still."))
        elif score["completeness"] < 0.9:
            warnings.append(advice.warn(
                f"the single-frame detector found {100 * score['completeness']:.0f} "
                "per cent of what the stack says is there. The rest are real "
                "cosmic rays it did not catch, and they are still in the "
                "cleaned frames.",
                f"lower --sigclip (now {params.sigclip}), or combine the "
                "frames with rejection instead, which uses the same "
                "information the stack just used."))
        if score["static"] and score["false_positive"] > 0.3:
            warnings.append(advice.warn(
                f"{100 * score['false_positive']:.0f} per cent of the "
                "detections are not supported by the other frames. Some of "
                "that is the stack missing faint hits; a large share is real "
                "structure being flagged.",
                f"raise --objlim (now {params.objlim}) so an edge has to be "
                "more unlike a star before it counts, or check the frames are "
                "really aligned."))

    def _write(self, params, spec, data, header, found, output, fits, np):
        """The frame -- cleaned, or as it was -- with the hits in its DQ, and
        their mask on its own when --crmask asks for one."""
        from ..core import dq, mef

        name = Path(spec.path).name
        stem = name[:-5] if name.endswith(".fits") else name
        _, _, var, quality = mef.load(spec)
        hit = np.asarray(found.mask, dtype=bool)
        quality = quality | np.where(hit, dq.COSMIC_RAY, 0).astype(dq.DTYPE)
        out = data
        if params.clean:
            out = found.cleaned
            if var is not None:
                var = _local_variance(var, hit)
        header = header.copy()
        header["HRWDCRM"] = (found.count, "pixels flagged as cosmic rays (DQ bit 8)")
        header.add_history(f"cosmicrays: {params.algorithm}, {found.count} pixels "
                           + ("replaced" if params.clean else "flagged"))

        written = []
        if params.crmask:
            mask = _destination(params.crmask, stem, ".mask.fits", spec, output,
                                suffix=".mask.fits")
            hdu = fits.PrimaryHDU(hit.astype(np.uint8))
            hdu.header["HRWDCRM"] = (found.count, "pixels flagged as cosmic rays")
            hdu.header.add_history(
                "cosmicrays: 1 where the pixel is no longer a measurement")
            hdu.writeto(mask, overwrite=True)
            header["CRMASK"] = (Path(mask).name, "the cosmic rays, as a mask image")
            written.append(Path(mask))
        destination = _destination(params.output, stem, ".fits", spec, output)
        files, notes = mef.save(destination, out, header, var, quality,
                                params.uncertainty, params.format)
        return files + written, notes


def _noise(params, header):
    """The gain and read noise to detect with: the parameters, as numbers or
    as keywords, else the frames' GAIN and RDNOISE, else 1 and 0 -- and a
    word that they were assumed. Returns the parameters with numbers in."""
    said = []

    def value(given, keyword, fallback):
        if given is not None:
            try:
                return float(given)
            except ValueError:
                if given in header:
                    return float(header[given])
                advice.fail(f"'{given}' is neither a number nor a keyword in the "
                            "header.", f"give the {keyword.lower()} as a number")
        if keyword in header:
            return float(header[keyword])
        said.append(advice.note(
            f"no {keyword} in the header and none given, so {fallback:g} was used. "
            "The noise model is what decides what counts as a cosmic ray.",
            f"give --{'gain' if keyword == 'GAIN' else 'rdnoise'}, from the "
            "detector's sheet."))
        return fallback

    gain = value(params.gain, "GAIN", 1.0)
    rdnoise = value(params.rdnoise, "RDNOISE", 0.0)
    return params.model_copy(update={"gain": gain, "rdnoise": rdnoise}), said


def _local_variance(var, hit):
    """What a replaced pixel's variance becomes: the median of its unflagged
    neighbours' in a 5x5 box -- the noise a pixel there would have had, as
    the value is now the median of theirs. The DQ keeps it flagged."""
    import numpy as np

    out = np.array(var, dtype=float)
    padded = np.pad(np.where(hit, np.nan, out), 2, constant_values=np.nan)
    for y, x in zip(*np.nonzero(hit), strict=True):
        box = padded[y:y + 5, x:x + 5]
        if np.isfinite(box).any():
            out[y, x] = float(np.nanmedian(box))
    return out


def _header_dispaxis(header) -> int | None:
    """The dispersion axis a frame declares, if it declares one."""
    value = header.get("DISPAXIS")
    try:
        axis = int(value)
    except (TypeError, ValueError):
        return None
    return axis if axis in (1, 2) else None


def _destination(where: str, stem: str, extension: str, spec, output,
                 suffix: str = ".fits") -> Path:
    """Where a result goes: a directory, a prefix, or this run's own folder."""
    if not where:
        return output.named_for(str(spec.path), suffix)
    path = Path(where)
    if path.is_dir() or where.endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{stem}{extension}"
    return Path(f"{where}{stem}{extension}")
