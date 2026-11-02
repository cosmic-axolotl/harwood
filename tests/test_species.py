"""Naming the atom, ion or molecule behind a feature.

The catalogue is curated rather than complete, and most of what is worth
testing here is the refusal: that an ambiguous wavelength is reported as
ambiguous, that a feature with no catalogue entry is reported as unnamed
rather than assigned to the nearest thing, and that a radial velocity moves
the answer.
"""

import numpy as np
import pytest

from harwood.core import species


def test_the_catalogue_loads_and_is_sorted():
    lines = species.catalogue()
    assert len(lines) > 100
    assert all(a.wavelength <= b.wavelength
               for a, b in zip(lines, lines[1:], strict=False))


def test_every_entry_says_what_produces_it():
    """The origin is usually more useful than the wavelength: a 7594
    absorption is not in the object at all."""
    for line in species.catalogue():
        assert line.origin, line.species
        assert all(origin in species.ORIGINS for origin in line.origin), line.species
        assert line.kind in species.KINDS, line.species
        assert 1 <= line.prominence <= 3, line.species


def test_the_obvious_lines_are_named_without_hesitation():
    for wavelength, expected in ((6562.8, "H I"), (4861.35, "H I"),
                                 (3933.66, "Ca II"), (5006.84, "[O III]"),
                                 (5889.95, "Na I"), (4685.68, "He II")):
        match = species.at(wavelength, tolerance=0.5)
        assert match.best is not None, wavelength
        assert match.best.line.species == expected, wavelength
        assert match.confident, f"{wavelength} should not be ambiguous"


def test_the_lithium_line_admits_its_iron_blend():
    """The real difficulty of the measurement, not hidden by a confident
    answer: Fe I 6707.44 is three tenths of an Angstrom away and is why every
    published lithium abundance has had to subtract something."""
    match = species.at(6707.8, tolerance=1.0)
    assert match.best.line.species == "Li I"
    assert not match.confident
    names = [candidate.line.species for candidate in match.candidates]
    assert "Fe I" in names


def test_a_feature_with_nothing_near_it_is_left_unnamed():
    """Reporting 'unidentified' is the point of a curated catalogue. A
    complete line list would answer this confidently and be useless."""
    match = species.at(6000.0, tolerance=0.5)
    assert match.best is None
    assert "not in the catalogue" in match.describe()


def test_prominence_decides_between_a_close_line_and_an_important_one():
    """H-alpha sitting almost on the measurement beats a faint iron line a
    little nearer; a faint line sitting exactly on it beats H-alpha an
    Angstrom away."""
    on_halpha = species.at(6562.9, tolerance=2.0)
    assert on_halpha.best.line.label == "H-alpha"

    on_iron = species.at(6707.44, tolerance=0.6)
    assert on_iron.best.line.species == "Fe I", "closeness wins when it is decisive"


def test_a_radial_velocity_moves_the_identification():
    """A star at 200 km/s puts H-alpha at 6567.2. Without the shift the line
    is unnamed; with it, it is hydrogen."""
    observed = species.shifted(6562.80, 200.0)
    assert observed == pytest.approx(6567.18, abs=0.02)

    assert species.at(observed, tolerance=0.5).best is None
    match = species.at(observed, tolerance=0.5, velocity=200.0)
    assert match.best.line.label == "H-alpha"
    assert abs(match.best.offset) < 0.05


def test_the_implied_velocity_is_reported_per_candidate():
    """'If this is it, the object is moving at X' -- which is how a wrong
    identification gives itself away."""
    match = species.at(6562.80 + 2.0, tolerance=3.0)
    halpha, = [item for item in match.candidates if item.line.label == "H-alpha"]
    assert halpha.velocity == pytest.approx(91.4, abs=1.0)


def test_narrowing_by_origin_is_how_outside_knowledge_enters():
    """The spectrum cannot say whether 6716.4 is nebular sulphur or the
    calcium line next to it. What you pointed at can."""
    both = species.at(6716.5, tolerance=1.5)
    assert len({item.line.species for item in both.candidates}) > 1

    nebula = species.at(6716.5, tolerance=1.5, origin="nebular")
    assert nebula.best.line.species == "[S II]"
    star = species.at(6716.5, tolerance=1.5, origin="stellar")
    assert star.best.line.species == "Ca I"


def test_air_and_vacuum_convert_back_to_each_other():
    """Mixing the conventions is a 2 Angstrom error in the red, invisible as
    anything but a wrong identification."""
    # The two directions are separate empirical fits rather than exact
    # inverses, so the round trip closes to a fraction of a milliangstrom --
    # far below anything that could change an identification.
    for air in (3500.0, 5000.0, 6562.8, 9000.0):
        assert species.vacuum_to_air(species.air_to_vacuum(air)) == \
            pytest.approx(air, abs=1e-3)
    assert species.air_to_vacuum(6562.80) - 6562.80 == pytest.approx(1.81, abs=0.05)


def test_below_two_thousand_angstroms_nothing_is_converted():
    """The conventions swap there because air does not transmit; converting
    would be applying a correction twice."""
    assert species.air_to_vacuum(1500.0) == 1500.0
    assert species.vacuum_to_air(1500.0) == 1500.0


def test_a_range_gives_everything_worth_marking():
    found = species.within(6500.0, 6600.0)
    assert any(line.label == "H-alpha" for line in found)
    assert all(6500 <= line.wavelength <= 6600 for line in found)

    strong = species.within(6500.0, 6600.0, minimum=3)
    assert len(strong) < len(found)


def test_a_species_can_be_found_by_name_or_by_words():
    assert species.named("Li I")[0].wavelength == pytest.approx(6103.64)
    assert species.named("lithium")
    assert species.named("nothing at all") == []


def test_agreeing_velocities_are_a_measurement_and_scattered_ones_are_not():
    # The tolerance has to be wide enough to reach across the shift, which is
    # 2.6 A at H-alpha: the point is that the shift is *discovered*, not that
    # it was assumed.
    real = [species.at(species.shifted(rest, 120.0), 4.0)
            for rest in (6562.80, 4861.35, 4340.47, 3933.66)]
    assert all(match.confident for match in real)
    median, scatter = species.velocity_from(real)
    assert median == pytest.approx(120.0, abs=5.0)
    assert scatter < 5.0


def test_telluric_lines_are_left_out_of_the_velocity():
    """They are in the air above the telescope, so they do not share the
    object's motion -- which makes them the check rather than part of the
    measurement."""
    matches = [species.at(species.shifted(rest, 150.0), 5.0)
               for rest in (6562.80, 4861.35, 4340.47)]
    matches.append(species.at(7593.7, 5.0))       # the O2 A band, unmoved
    assert matches[-1].best.line.species == "O2"

    median, scatter = species.velocity_from(matches)
    assert median == pytest.approx(150.0, abs=5.0)
    assert scatter < 5.0, "the telluric band must not widen the scatter"


def test_too_few_lines_gives_no_velocity_rather_than_a_bad_one():
    assert species.velocity_from([species.at(6562.8, 1.0)]) is None


# --------------------------------------------------------------- the task


@pytest.fixture
def calibrated_star(tmp_path, monkeypatch):
    """A synthetic stellar spectrum with lines put in at known wavelengths."""
    from harwood.core import spectrum

    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(3)

    wavelength = np.linspace(6500.0, 6800.0, 1500)
    flux = np.full(wavelength.size, 1000.0)
    # Keyed by the name the catalogue gives each line: a line with no label
    # of its own is named by its species and wavelength.
    put_in = {"H-alpha": 6562.80, "He I 6678.15": 6678.15, "DIB 6614": 6613.6,
              "[S II] 6716": 6716.44, "Li I 6708": 6707.76}
    for centre in put_in.values():
        flux -= 400.0 * np.exp(-0.5 * ((wavelength - centre) / 0.9) ** 2)

    ap = spectrum.Aperture(
        number=1, centre=50.0, low=-8.0, high=8.0,
        pixel=np.arange(1.0, wavelength.size + 1.0),
        wavelength=wavelength,
        flux=flux + rng.normal(0, 4, wavelength.size),
        error=np.full(wavelength.size, 4.0), title="a made-up star")
    spectrum.write(tmp_path / "star.fits", [ap], source="synthetic")
    return tmp_path, put_in


def test_lineid_names_the_lines_that_were_put_in(calibrated_star):
    from harwood.core.registry import get_task

    work, put_in = calibrated_star
    result = get_task("lineid").run(input="star.fits", output=work / "out.csv",
                                    features="absorption", tolerance=1.0)

    named = {row["line"] for row in result.data["rows"]}
    for label in put_in:
        assert label in named, f"{label} was put in and should come back"


def test_lineid_gets_the_wavelengths_right_to_a_fraction_of_a_pixel(calibrated_star):
    from harwood.core.registry import get_task

    work, put_in = calibrated_star
    result = get_task("lineid").run(input="star.fits", output=work / "out.csv",
                                    features="absorption")

    by_line = {row["line"]: row["wavelength"] for row in result.data["rows"]}
    for label, truth in put_in.items():
        assert by_line[label] == pytest.approx(truth, abs=0.3), label


def test_lineid_refuses_a_spectrum_that_is_still_in_pixels(tmp_path, monkeypatch):
    """Naming a line needs to know what wavelength it is at, and saying so is
    more use than naming whatever happens to sit at pixel 512."""
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    ap = spectrum.Aperture(number=1, centre=1.0, low=-1.0, high=1.0,
                           pixel=np.arange(1.0, 101.0), flux=np.ones(100),
                           error=np.ones(100))
    spectrum.write(tmp_path / "raw.fits", [ap])

    with pytest.raises(ValueError) as raised:
        get_task("lineid").run(input="raw.fits", output=tmp_path / "x.csv")
    assert "dispcor" in raised.value.fix


def test_lineid_says_when_the_object_is_shifted(calibrated_star):
    """Every line moved by the same fraction: that is a radial velocity, and
    the task should notice rather than silently failing to identify."""
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    work, _ = calibrated_star
    star = spectrum.read(work / "star.fits")[0]
    star.wavelength = species.shifted(star.wavelength, 150.0)
    spectrum.write(work / "moving.fits", [star])

    result = get_task("lineid").run(input="moving.fits", output=work / "m.csv",
                                    features="absorption", tolerance=4.0)

    assert any("--velocity" in warning for warning in result.warnings)
    step, = [item for item in result.steps if item.action == "velocity"]
    assert float(step.measured["implied shift"].split()[0]) == pytest.approx(
        150.0, abs=10.0)


def test_lines_a_pixel_apart_at_rest_are_not_called_misidentified(tmp_path, monkeypatch):
    """A star at rest whose line centres scatter by a pixel is a measurement,
    not 'a set of misidentifications': the check used to compare the scatter
    with the shift itself, and a shift near zero made any scatter a problem.
    (DH Cep's He II, two minima 0.7 A apart, was flagged that way.)"""
    from harwood.core import spectrum
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    wavelength = np.linspace(6500.0, 6800.0, 1500)            # 0.2 A, 9 km/s a pixel
    flux = np.full(wavelength.size, 1000.0)
    # Two of the confident lines a pixel to the red, two to the blue: the
    # shift is zero, the scatter a pixel. (Li I is ambiguous and stays out.)
    for centre, nudge in ((6562.80, 0.2), (6678.15, -0.2), (6613.6, -0.2),
                          (6716.44, 0.2), (6707.76, 0.0)):
        flux -= 400.0 * np.exp(-0.5 * ((wavelength - centre - nudge) / 0.9) ** 2)
    ap = spectrum.Aperture(number=1, centre=50.0, low=-8.0, high=8.0,
                           pixel=np.arange(1.0, wavelength.size + 1.0),
                           wavelength=wavelength, flux=flux,
                           error=np.full(wavelength.size, 4.0))
    spectrum.write(tmp_path / "rest.fits", [ap])
    result = get_task("lineid").run(input="rest.fits", output=tmp_path / "r.csv",
                                    features="absorption", tolerance=1.0)
    step, = [item for item in result.steps if item.action == "velocity"]
    assert float(step.measured["scatter"].split()[0]) > 3.0, "it does scatter"
    assert not any("do not agree on a shift" in str(w) for w in result.warnings)


def test_lineid_writes_a_table_that_explains_its_own_columns(calibrated_star):
    from harwood.core.registry import get_task

    work, _ = calibrated_star
    get_task("lineid").run(input="star.fits", output=work / "out.csv",
                           features="absorption")

    text = (work / "out.csv").read_text(encoding="utf-8")
    assert "air wavelengths" in text
    assert "wavelength,kind,species,line" in text
    # The prose about each species is carried into the file, so the table is
    # readable without the program that wrote it.
    assert "lithium" in text.lower()


def test_lineid_can_be_narrowed_to_one_origin(calibrated_star):
    from harwood.core.registry import get_task

    work, _ = calibrated_star
    result = get_task("lineid").run(input="star.fits", output=work / "n.csv",
                                    features="absorption", origin="interstellar")

    named = {row["species"] for row in result.data["rows"] if row["species"] != "-"}
    assert named == {"DIB"}, "only the interstellar band should be named"


def test_an_unknown_origin_says_which_ones_exist(calibrated_star):
    from harwood.core.registry import get_task

    work, _ = calibrated_star
    with pytest.raises(ValueError) as raised:
        get_task("lineid").run(input="star.fits", output=work / "x.csv",
                               origin="galactic")
    assert "telluric" in raised.value.fix


# ------------------------------------------- naming as a check on dispcor


def _star(wavelength, seed=5):
    """A synthetic stellar spectrum on a given wavelength scale.

    With noise, and not for realism: the detection estimates its threshold
    from the scatter between the lines, so a perfectly noiseless spectrum
    makes it fall back to a standard deviation the lines themselves dominate,
    and only the deepest feature survives.
    """
    from harwood.core import spectrum

    flux = np.full(wavelength.size, 2000.0)
    for centre, depth in ((6562.80, 900), (6678.15, 500), (6707.76, 300),
                          (6716.44, 250), (6613.6, 200)):
        flux -= depth * np.exp(-0.5 * ((wavelength - centre) / 0.8) ** 2)
    flux += np.random.default_rng(seed).normal(0, 8, wavelength.size)
    return spectrum.Aperture(
        number=1, centre=50.0, low=-8.0, high=8.0,
        pixel=np.arange(1.0, wavelength.size + 1.0), wavelength=wavelength,
        flux=flux, error=np.full(wavelength.size, 8.0))


def test_a_right_solution_puts_the_features_on_real_lines():
    from harwood.tasks.onedspec import _recognise

    wave = np.linspace(6550.0, 6750.0, 1024)
    named, unnamed = _recognise(_star(wave))

    assert "H-alpha" in named and "He I 6678.15" in named
    assert not unnamed, "every strong feature should land on something"


def test_a_solution_off_by_five_angstroms_is_caught_by_the_names():
    """The point of checking against the catalogue rather than against the arc.

    A dispersion solution fitted to the wrong catalogue lines reproduces those
    lines beautifully -- the rms says nothing is wrong. Real features landing
    on real wavelengths is a separate claim, and it is the one that fails here.
    """
    from harwood.tasks.onedspec import _recognise

    truth = np.linspace(6550.0, 6750.0, 1024)
    star = _star(truth)
    star.wavelength = truth + 5.0          # the solution is five Angstroms out

    named, unnamed = _recognise(star)
    assert unnamed, "the features no longer land on anything"
    assert "H-alpha" not in named


def test_dispcor_says_what_it_recognised(flexing_night_star):
    from harwood.core.registry import get_task

    work = flexing_night_star
    result = get_task("dispcor").run(input="star.ms.fits", output="w.fits",
                                     reference="early", database=work)

    check, = [step for step in result.steps if step.action == "check"]
    assert "landing on" in check.measured


@pytest.fixture
def flexing_night_star(tmp_path, monkeypatch):
    """An arc with a solution in the database, and a star to calibrate."""
    from astropy.io import fits

    from harwood.core import arc, spectrum
    from harwood.core.registry import get_task

    if not any(d.is_dir() for d in arc.IRAF_LINELISTS):
        pytest.skip("IRAF line lists are not installed")

    monkeypatch.chdir(tmp_path)
    crval, cdelt, npixels = 6650.0, 0.152, 1024
    centre = (npixels + 1) / 2
    catalogue = arc.read_linelist("linelists$thorium.dat")[0]

    rng = np.random.default_rng(11)
    low = crval + (1 - centre) * cdelt
    high = crval + (npixels - centre) * cdelt
    strong = catalogue[(catalogue > low) & (catalogue < high)]
    strong = strong[rng.random(strong.size) < 0.5]

    flux = np.full(npixels, 300.0)
    grid = np.arange(1.0, npixels + 1.0)
    for position, amplitude in zip((strong - crval) / cdelt + centre,
                                   10 ** rng.uniform(2.5, 4.3, strong.size),
                                   strict=False):
        flux += amplitude * np.exp(-0.5 * ((grid - position) / 0.72) ** 2)
    fits.PrimaryHDU((flux + rng.normal(0, 12, npixels)).astype(np.float32)).writeto(
        tmp_path / "early.fits")
    get_task("identify").run(image="early.fits", crval=crval, cdelt=cdelt,
                             coordlist="linelists$thorium.dat", function="chebyshev",
                             order=3, niterate=3, database=tmp_path)

    wave = crval + (grid - centre) * cdelt
    spectrum.write(tmp_path / "star.ms.fits", [_star(wave)])
    return tmp_path


# --------------------------------------------- what the panel would draw


def test_marks_move_with_the_object_but_the_sky_stays_put():
    """The mistake this prevents: drawing a telluric band shifted along with
    the star makes a wrong radial velocity look right, because everything on
    screen lines up."""
    at_rest = {mark.line.name: mark.at for mark in species.marks(6500.0, 7700.0)}
    moving = {mark.line.name: mark.at
              for mark in species.marks(6500.0, 7700.0, velocity=300.0)}

    assert moving["H-alpha"] - at_rest["H-alpha"] == pytest.approx(6.57, abs=0.05)
    assert moving["O2 A band"] == at_rest["O2 A band"], "the air does not move"


def test_a_line_shifted_into_range_is_still_drawn():
    """The range has to be widened before the shift, or a line just outside
    the spectrum whose shifted position falls inside it is never marked."""
    below = 6562.80 - 0.5
    assert not any(mark.line.label == "H-alpha"
                   for mark in species.marks(below, 6600.0, velocity=-300.0))
    assert any(mark.line.label == "H-alpha"
               for mark in species.marks(below, 6600.0, velocity=+300.0))


def test_marks_can_be_narrowed_to_the_origins_being_shown():
    every = species.marks(6500.0, 7700.0)
    sky_only = species.marks(6500.0, 7700.0, origins={"telluric"})

    assert len(sky_only) < len(every)
    assert all("telluric" in mark.line.origin for mark in sky_only)


def test_a_mark_knows_whether_a_real_feature_sits_on_it():
    """What the panel's `f` key filters on: every line in range is context,
    but the ones with a measured feature under them are the findings."""
    marks = species.marks(6500.0, 6800.0, measured=[6562.9], tolerance=1.0)
    by_name = {mark.line.name: mark.detected for mark in marks}

    assert by_name["H-alpha"] is True
    assert by_name["DIB 6614"] is False


def test_every_mark_has_a_colour_key_the_panel_knows():
    """The panel is optional -- harwood installs without Qt -- but when it is
    there, no origin may fall through to a default colour."""
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from harwood.viewer.species_panel import CATEGORIES

    for mark in species.marks(3500.0, 9000.0, minimum=1):
        assert mark.colour_key in CATEGORIES, mark.line.species
