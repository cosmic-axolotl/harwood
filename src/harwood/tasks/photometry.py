"""Photometry -- detection and aperture measurement.

IRAF names throughout, including the ones that are not obvious: the task is
`phot`, not `apphot` (`apphot` is the *package*); the aperture radii are
`apertures`; the sky annulus is `annulus` and `dannulus`; the gain is `epadu`;
the magnitude zero point is `zmag`. Section 3 is explicit that a different name
costs little and the same name meaning something else ruins a result.

Every coordinate that leaves this module is 1-based and (x, y) -- section 2.2.
`photutils` works 0-based, so the conversion happens at the call, once, through
`core.pixels`. The catalogue written here can be read by someone who has spent
twenty years reading IRAF `.mag` files without a systematic pixel offset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult


class DaoFindParams(TaskParams):
    image: str = Field(..., description="Image to search. Accepts 'obj.fits[SCI,1]'.")
    output: Path = Field(..., description="Coordinate catalogue to write (.ecsv).")

    fwhm: float = Field(
        2.5, gt=0,
        description="FWHM of the point spread function, in pixels. The single most "
                    "important parameter: measure it with imexamine before guessing.",
    )
    sigma: float | None = Field(
        None, gt=0,
        description="Standard deviation of the background. Measured from the image "
                    "when not given.",
    )
    threshold: float = Field(
        4.0, gt=0, description="Detection threshold above background, in units of sigma."
    )
    sharplo: float = Field(0.2, description="Lower sharpness bound; rejects cosmic rays.")
    sharphi: float = Field(1.0, description="Upper sharpness bound; rejects galaxies.")
    roundlo: float = Field(-1.0, description="Lower roundness bound.")
    roundhi: float = Field(1.0, description="Upper roundness bound.")
    border: float | None = Field(
        None, ge=0,
        description="Ignore detections within this many pixels of the edge. "
                    "Defaults to twice the FWHM. The detection filter needs whole "
                    "pixels around a source to judge it, so anything closer to the "
                    "edge than its own width is an artefact of the truncated "
                    "convolution, not a star. Set 0 to keep them.",
    )


@register
class DaoFind(Task[DaoFindParams]):
    name = "daofind"
    package = "apphot"
    description = "Detect point sources with the DAOFIND algorithm."
    Params = DaoFindParams

    def plan(self, params: DaoFindParams) -> list[Step]:
        return [
            Step("read", params.image),
            Step("background", "sigma-clipped statistics"
                 if params.sigma is None else f"sigma={params.sigma} as given"),
            Step("detect", f"DAOFIND, fwhm={params.fwhm}, "
                           f"threshold={params.threshold} sigma"),
            Step("filter", f"sharpness {params.sharplo}..{params.sharphi}, "
                           f"roundness {params.roundlo}..{params.roundhi}"),
            Step("border", f"drop anything within "
                           f"{_border(params):g} px of the edge"),
            Step("write", f"{params.output}, coordinates 1-based (x,y)"),
        ]

    def execute(self, params: DaoFindParams) -> TaskResult:
        import numpy as np
        from astropy.stats import sigma_clipped_stats
        from astropy.table import Table
        from photutils.detection import DAOStarFinder

        from ..core import catalog, images, pixels, provenance

        data, header = images.read(params.image)
        mean, median, sigma = sigma_clipped_stats(data, sigma=3.0)
        if params.sigma is not None:
            sigma = params.sigma

        finder = DAOStarFinder(fwhm=params.fwhm, threshold=params.threshold * sigma,
                               **_shape_bounds(DAOStarFinder, params))
        found = finder(data - median)
        if found is None or len(found) == 0:
            raise RuntimeError(
                f"no sources found. The background sigma measured {sigma:.2f}; try a "
                f"lower --threshold than {params.threshold}, or check --fwhm."
            )

        # The one conversion. photutils counts from 0; everything harwood
        # writes counts from 1.
        position = pixels.from_numpy(_column(found, "y_centroid", "ycentroid"),
                                     _column(found, "x_centroid", "xcentroid"))
        border = _border(params)
        ny, nx = data.shape
        inside = ((position.x > border) & (position.x <= nx - border)
                  & (position.y > border) & (position.y <= ny - border))
        dropped = int((~inside).sum())
        if not inside.any():
            raise RuntimeError(
                f"every one of the {len(found)} detections was within {border:g} px "
                "of the edge. Lower --border if the field really does run to the edge."
            )
        found, position = found[inside], pixels.Position(position.x[inside],
                                                         position.y[inside])
        # Brightest first. It is the one ordering two frames of the same field
        # agree on, which is what xyxymatch's triangle search needs when it
        # takes only the brightest nmatch stars from each list.
        order = np.argsort(np.asarray(_column(found, "flux")))[::-1]
        found, position = found[order], pixels.Position(position.x[order],
                                                        position.y[order])
        catalogue = Table({
            "id": range(1, len(found) + 1),
            "x": position.x,
            "y": position.y,
            "sharpness": _column(found, "sharpness"),
            "roundness": _column(found, "roundness1", "roundness"),
            "peak": _column(found, "peak", "max_value"),
            "flux": _column(found, "flux"),
        })
        catalog.write(catalogue, params.output,
                      harwood_image=str(params.image),
                      harwood_background=float(median),
                      harwood_sigma=float(sigma),
                      harwood_call=provenance.current_stamp() or "")

        return TaskResult(
            outputs=[params.output],
            data={"sources": len(catalogue), "dropped_at_border": dropped,
                  "background": float(median), "sigma": float(sigma)},
            history=[f"daofind: {len(catalogue)} sources, fwhm={params.fwhm}, "
                     f"threshold={params.threshold} sigma (sky={median:.2f}, "
                     f"sigma={sigma:.2f}), {dropped} dropped within "
                     f"{border:g} px of the edge"],
            steps=self.plan(params),
        )


class PhotParams(TaskParams):
    image: str = Field(..., description="Image to measure.")
    coords: Path = Field(..., description="Coordinate catalogue, 1-based (x,y), from daofind.")
    output: Path = Field(..., description="Photometry table to write (.ecsv).")

    apertures: list[float] = Field(
        [5.0], description="Aperture radii in pixels. Several may be given."
    )
    annulus: float = Field(
        10.0, gt=0, description="Inner radius of the sky annulus, in pixels."
    )
    dannulus: float = Field(5.0, gt=0, description="Width of the sky annulus, in pixels.")
    salgorithm: Literal["median", "mean", "mode"] = Field(
        "median", description="How the sky level is estimated inside the annulus."
    )

    zmag: float = Field(25.0, description="Zero point of the instrumental magnitude scale.")
    epadu: float = Field(
        1.0, gt=0, description="Gain in electrons per ADU, for the Poisson error term."
    )
    readnoise: float = Field(0.0, ge=0, description="Read noise in electrons, for the error.")
    exposure: str = Field(
        "EXPTIME", description="Header keyword holding the exposure time."
    )

    @model_validator(mode="after")
    def _check_radii(self) -> PhotParams:
        if max(self.apertures) >= self.annulus:
            raise ValueError(
                f"the sky annulus starts at {self.annulus} px but the largest aperture "
                f"is {max(self.apertures)} px: the sky would be measured over the star. "
                "Raise --annulus or lower --apertures."
            )
        return self


@register
class Phot(Task[PhotParams]):
    name = "phot"
    package = "apphot"
    description = "Aperture photometry with a local sky annulus."
    Params = PhotParams

    def plan(self, params: PhotParams) -> list[Step]:
        radii = ", ".join(str(r) for r in params.apertures)
        return [
            Step("read", f"{params.image} and {params.coords}"),
            Step("sky", f"{params.salgorithm} over the annulus "
                        f"{params.annulus}..{params.annulus + params.dannulus} px, "
                        f"with its measured scatter for the error"),
            Step("measure", f"apertures {radii} px"),
            Step("calibrate", f"zmag={params.zmag}, epadu={params.epadu}, "
                              f"exposure from {params.exposure}"),
            Step("write", str(params.output)),
        ]

    def execute(self, params: PhotParams) -> TaskResult:
        import numpy as np
        from astropy.table import Table
        from photutils.aperture import (
            ApertureStats,
            CircularAnnulus,
            CircularAperture,
            aperture_photometry,
        )

        from ..core import catalog, images, pixels, provenance

        data, header = images.read(params.image)
        exptime = float(header.get(params.exposure, 1.0))
        catalogue = catalog.read(params.coords)
        x, y = catalog.coordinates(catalogue, params.coords)

        # 1-based (x, y) in the file -> 0-based (x, y) for photutils.
        row, col = pixels.to_numpy(x, y)
        positions = np.transpose((col, row))

        annulus = CircularAnnulus(positions, r_in=params.annulus,
                                  r_out=params.annulus + params.dannulus)
        stats = ApertureStats(data, annulus)
        sky = np.asarray(getattr(stats, params.salgorithm), dtype=float)
        # The *measured* scatter of the sky annulus, not the Poisson value of
        # its level: in a crowded field, flat-field residuals and the wings of
        # neighbours dominate, and a theoretical sky sigma understates the real
        # error by an order of magnitude. This is IRAF's `stdev`.
        sky_sigma = np.asarray(stats.std, dtype=float)
        nsky = np.asarray(stats.sum_aper_area.value if hasattr(stats.sum_aper_area,
                                                               "value")
                          else stats.sum_aper_area, dtype=float)

        out = Table({"id": catalogue["id"] if "id" in catalogue.colnames
                     else range(1, len(x) + 1), "x": x, "y": y,
                     "msky": sky, "stdev": sky_sigma, "nsky": nsky})
        for radius in params.apertures:
            aperture = CircularAperture(positions, r=radius)
            raw = aperture_photometry(data, aperture)["aperture_sum"]
            area = aperture.area_overlap(data)
            flux = np.asarray(raw, dtype=float) - sky * area

            # IRAF's apphot error, in ADU^2:
            #   flux/epadu          photon noise from the star
            #   area * stdev^2      sky noise over the aperture
            #   area^2 * stdev^2/nsky   uncertainty in the mean sky itself
            # The third term is why a small sky annulus costs you accuracy on
            # faint stars, and it is the term a naive implementation drops.
            with np.errstate(divide="ignore", invalid="ignore"):
                variance = (np.clip(flux, 0, None) / params.epadu
                            + area * sky_sigma ** 2
                            + np.divide(area ** 2 * sky_sigma ** 2, nsky,
                                        out=np.zeros_like(area, dtype=float),
                                        where=nsky > 0))
            if params.readnoise > 0:
                variance = variance + area * (params.readnoise / params.epadu) ** 2
            flux_error = np.sqrt(variance)

            with np.errstate(divide="ignore", invalid="ignore"):
                mag = params.zmag - 2.5 * np.log10(np.where(flux > 0, flux, np.nan)
                                                   / exptime)
                merr = 1.0857 * np.where(flux > 0, flux_error / flux, np.nan)
            suffix = "" if len(params.apertures) == 1 else f"_{radius:g}"
            out[f"flux{suffix}"] = flux
            out[f"mag{suffix}"] = mag
            out[f"merr{suffix}"] = merr

        catalog.write(out, params.output,
                      harwood_image=str(params.image),
                      harwood_exposure=exptime,
                      harwood_zmag=params.zmag,
                      harwood_call=provenance.current_stamp() or "")

        first = "" if len(params.apertures) == 1 else f"_{params.apertures[0]:g}"
        measured = int(np.isfinite(out[f"mag{first}"]).sum())
        return TaskResult(
            outputs=[params.output],
            data={"stars": len(out), "measured": measured,
                  "sky_median": float(np.median(sky))},
            history=[f"phot: apertures={params.apertures} annulus={params.annulus} "
                     f"dannulus={params.dannulus} zmag={params.zmag} epadu={params.epadu}"],
            steps=self.plan(params),
        )


def _border(params) -> float:
    """How close to the edge a detection may sit.

    IRAF's daofind skips the border because its convolution kernel does not
    fit there; photutils will happily report the artefact. On the M92 frames
    a fifth of all detections came from the truncated edge, so the default
    matters more than it looks.
    """
    return float(params.border) if params.border is not None else 2.0 * params.fwhm


def _shape_bounds(finder_class, params) -> dict:
    """Pass the sharpness and roundness limits under whichever names apply.

    photutils 3.0 replaced `sharplo`/`sharphi` with `sharpness_range`. harwood
    keeps IRAF's four names on its own surface -- they are what a user types --
    and adapts here, at the boundary, where the ecosystem's churn belongs.
    """
    import inspect

    accepted = inspect.signature(finder_class).parameters
    if "sharpness_range" in accepted:
        return {"sharpness_range": (params.sharplo, params.sharphi),
                "roundness_range": (params.roundlo, params.roundhi)}
    return {"sharplo": params.sharplo, "sharphi": params.sharphi,
            "roundlo": params.roundlo, "roundhi": params.roundhi}


def _column(table, *names):
    """First column that exists, so a photutils rename is not an outage."""
    for name in names:
        if name in table.colnames:
            return table[name]
    raise KeyError(
        f"photutils returned none of {names}; it has {', '.join(table.colnames)}. "
        "This is a photutils API change -- harwood needs updating."
    )
