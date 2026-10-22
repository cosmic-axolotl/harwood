"""Core: parameter resolution, image specs, the registry, provenance."""


import numpy as np
import pytest
from astropy.io import fits
from pydantic import Field

from harwood.core import images, migration, params, provenance
from harwood.core.registry import get_task, list_tasks
from harwood.core.task import TaskParams, TaskResult


class DemoParams(TaskParams):
    input: str = Field(..., description="An input.")
    sigma: float = Field(3.0, ge=0, description="A threshold.")
    reject: str = Field("none", description="A choice.")


# ----------------------------------------------------------- parameters


def test_nothing_is_read_from_the_home_directory(tmp_path, monkeypatch):
    """The uparm$ ban of section 2, as an executable check."""
    monkeypatch.chdir(tmp_path)
    resolved = params.resolve("demo", DemoParams)
    assert all(r.origin == "default" for r in resolved.values())
    assert params.find_project() is None


def test_project_file_beats_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params.write_project(tmp_path / params.PROJECT_FILE, {"demo": {"sigma": 5.0}})
    resolved = params.resolve("demo", DemoParams)
    assert resolved["sigma"].value == 5.0
    assert resolved["sigma"].origin == "project"


def test_params_all_holds_for_every_task_that_has_the_parameter(tmp_path, monkeypatch):
    """How a reduction says once that it wants ERR, or IRAF's format: under
    params.all. A task's own entry still wins, and a task without the
    parameter is not bothered by it."""
    monkeypatch.chdir(tmp_path)
    params.write_project(tmp_path / params.PROJECT_FILE,
                         {"all": {"sigma": 6.0, "uncertainty": "ERR"},
                          "other": {"sigma": 2.0}})
    resolved = params.resolve("demo", DemoParams)
    assert resolved["sigma"].value == 6.0 and resolved["sigma"].origin == "project"
    assert "uncertainty" not in resolved
    assert params.resolve("other", DemoParams)["sigma"].value == 2.0
    ccdproc = params.resolve("ccdproc", get_task("ccdproc").Params)
    assert ccdproc["uncertainty"].value == "ERR"


def test_command_line_beats_everything(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params.write_project(tmp_path / params.PROJECT_FILE, {"demo": {"sigma": 5.0}})
    resolved = params.resolve("demo", DemoParams, overrides={"sigma": 9.0},
                              preset={"sigma": 7.0}, preset_name="demo2")
    assert resolved["sigma"].value == 9.0
    assert resolved["sigma"].origin == "command line"


def test_preset_sits_between_project_and_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = params.resolve("demo", DemoParams, preset={"reject": "minmax"},
                              preset_name="zerocombine")
    assert resolved["reject"].value == "minmax"
    assert "zerocombine" in resolved["reject"].describe()


def test_a_subdirectory_still_sees_the_reduction(tmp_path, monkeypatch):
    params.write_project(tmp_path / params.PROJECT_FILE, {"demo": {"sigma": 4.0}})
    deep = tmp_path / "night" / "raw"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert params.resolve("demo", DemoParams)["sigma"].value == 4.0


def test_a_typo_in_a_parameter_name_is_caught_by_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(KeyError, match="sgima"):
        params.resolve("demo", DemoParams, overrides={"sgima": 3.0})


# --------------------------------------------------------- image specs


@pytest.mark.parametrize("text,ext,section", [
    ("obj.fits", None, None),
    ("obj.fits[1]", 1, None),
    ("obj.fits[SCI,1]", ("SCI", 1), None),
    ("obj.fits[SCI,1][100:400,*]", ("SCI", 1), "100:400,*"),
])
def test_image_spec_parsing(text, ext, section):
    spec = images.parse(text)
    assert spec.extension == ext
    assert spec.section == section


def test_at_list_is_expanded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.fits").touch()
    (tmp_path / "b.fits").touch()
    (tmp_path / "list.lis").write_text("a.fits\n# a comment\nb.fits\n")
    assert [str(s) for s in images.expand("@list.lis")] == ["a.fits", "b.fits"]


def test_a_glob_that_matches_nothing_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="nope"):
        images.expand("nope*.fits")


def test_reading_a_section_gives_the_right_pixels(tmp_path):
    path = tmp_path / "x.fits"
    fits.PrimaryHDU(np.arange(100, dtype=float).reshape(10, 10)).writeto(path)
    data, header = images.read(f"{path}[2:4,1:1]")
    np.testing.assert_array_equal(data, [[1.0, 2.0, 3.0]])
    assert "HRWDSECT" in header


#: IRAF 2.18.1's imcopy of a 70x60 image whose pixel (x, y) holds
#: 70 (y - 1) + x - 1, with CRPIX 50 40 and CD [[-1e-4, 2e-5], [3e-5, 1e-4]]:
#: shape, first pixel, CRPIX1 CRPIX2, CD1_1 CD1_2 CD2_1 CD2_2, LTV1 LTV2,
#: LTM1_1 LTM2_2 (None where IRAF writes no card).
IRAF_SECTIONS = {
    "[6:60,3:50]": ((48, 55), 145.0, (45.0, 38.0), (-1e-4, 2e-5, 3e-5, 1e-4),
                    (-5.0, -2.0), (None, None)),
    "[*:2,*]": ((60, 35), 0.0, (25.5, 40.0), (-2e-4, 2e-5, 6e-5, 1e-4),
                (0.5, None), (0.5, None)),
    "[-*,*]": ((60, 70), 69.0, (21.0, 40.0), (1e-4, 2e-5, -3e-5, 1e-4),
               (71.0, None), (-1.0, None)),
    "[60:6,10:40:3]": ((11, 55), 689.0, (11.0, 11.0), (1e-4, 6e-5, -3e-5, 3e-4),
                       (61.0, -2.333333333333), (-1.0, 0.333333333333)),
    "[5:65:4,-*]": ((60, 16), 4134.0, (12.25, 21.0), (-4e-4, -2e-5, 1.2e-4, -1e-4),
                    (-0.25, 61.0), (0.25, -1.0)),
}


@pytest.mark.parametrize("section", list(IRAF_SECTIONS))
def test_a_section_moves_the_wcs_as_iraf_s_imcopy_does(tmp_path, section):
    shape, first, crpix, cd, ltv, ltm = IRAF_SECTIONS[section]
    header = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRPIX1": 50.0,
                          "CRPIX2": 40.0, "CRVAL1": 10.0, "CRVAL2": 20.0, "CD1_1": -1e-4,
                          "CD1_2": 2e-5, "CD2_1": 3e-5, "CD2_2": 1e-4})
    path = tmp_path / "w.fits"
    fits.writeto(path, np.arange(60 * 70, dtype=np.float32).reshape(60, 70), header)
    data, read = images.read(f"{path}{section}")
    assert data.shape == shape and data[0, 0] == first
    assert (read["CRPIX1"], read["CRPIX2"]) == pytest.approx(crpix)
    assert [read[k] for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")] == pytest.approx(cd)
    for axis in (1, 2):
        for key, expected in ((f"LTV{axis}", ltv[axis - 1]),
                              (f"LTM{axis}_{axis}", ltm[axis - 1])):
            if expected is None:
                assert key not in read
            else:
                assert read[key] == pytest.approx(expected)


def test_mismatched_shapes_are_refused_with_both_shapes(tmp_path):
    small, big = tmp_path / "s.fits", tmp_path / "b.fits"
    fits.PrimaryHDU(np.zeros((4, 4))).writeto(small)
    fits.PrimaryHDU(np.zeros((8, 8))).writeto(big)
    with pytest.raises(ValueError, match=r"\(8, 8\)"):
        images.read_stack([small, big])


# ------------------------------------------------------------ provenance


def test_the_log_records_origins_and_backend(tmp_path):
    result = TaskResult(outputs=[tmp_path / "out.fits"], backend="numba 0.67.0")
    provenance.record("demo", {"sigma": 3.0}, result, tmp_path,
                      origins={"sigma": "project: harwood.yaml"})
    entry = provenance.read_log(tmp_path)[-1]
    assert entry["origins"]["sigma"] == "project: harwood.yaml"
    assert entry["backend"] == "numba 0.67.0"


def test_header_cards_stay_ascii():
    """Section 6.2: FITS cards take no accents, whatever the interface speaks."""
    header = fits.Header()
    token = provenance.set_call_stamp("harwood.phot medição de fluxo")
    try:
        provenance.stamp_header(header)
    finally:
        provenance.reset_call_stamp(token)
    text = "".join(str(card) for card in header.cards)
    assert text.isascii()
    assert "HARWOOD" in header


def test_imcmb_cards_name_every_input():
    header = fits.Header()
    provenance.record_inputs(header, [f"bias{i:02d}.fits" for i in range(1, 6)])
    assert header["NCOMBINE"] == 5
    assert header["IMCMB003"] == "bias03.fits"


def test_above_999_inputs_the_keyword_gives_way_to_history():
    header = fits.Header()
    provenance.record_inputs(header, [f"f{i}.fits" for i in range(1500)])
    assert header["NCOMBINE"] == 1500
    assert "IMCMB001" not in header


# -------------------------------------------------------------- registry


def test_every_registered_task_has_described_parameters():
    """The parameter text is the help, the editor tooltip and the translation
    source all at once (section 5). A field without one is a hole in all three."""
    for name, cls in list_tasks().items():
        for field_name, field in cls.Params.model_fields.items():
            assert field.description, f"{name}.{field_name} has no description"


def test_the_migration_map_tells_the_truth_about_what_runs():
    """The map is what `harwood list --iraf` and `harwood help` answer from.

    A task that runs here but is listed as `planned` makes harwood lie about
    its own state, and someone reading the map to decide whether to migrate
    gets the wrong answer. This test is why the map cannot drift.
    """
    mapped = {e.get("harwood", n): (n, e) for n, e in migration.tasks().items()}
    mapped.update({n: (n, e) for n, e in migration.added().items()})
    for name in list_tasks():
        assert name in mapped, f"'{name}' is in neither IRAF nor the migration map"
        iraf_name, entry = mapped[name]
        assert entry["status"] in ("available", "partial"), (
            f"'{name}' runs, but the map says '{entry['status']}'"
        )
        if entry["status"] == "partial":
            assert entry.get("note"), f"'{name}' is partial without saying what is missing"


def test_a_task_listed_as_available_is_actually_registered():
    """The other direction: the map must not promise what does not exist."""
    from harwood.core.registry import exists

    for name, entry in migration.tasks().items():
        if entry["status"] in ("available", "partial"):
            assert exists(entry.get("harwood", name)), \
                f"the map claims '{name}' runs, but no task is registered for it"


def test_a_near_miss_suggests_the_real_name():
    with pytest.raises(KeyError, match="daofind"):
        get_task("daofnd")


def test_presets_are_the_same_engine():
    from harwood.tasks.imcombine import ImCombine

    zero = get_task("zerocombine")
    assert isinstance(zero, ImCombine)
    assert zero.defaults["reject"] == "minmax"


def test_migration_map_answers_for_a_task_harwood_does_not_run():
    assert "not be ported" in migration.explain("rfits")
    assert "after the MVP" in migration.explain("continuum")
    assert migration.explain("daofind") is None
    assert migration.explain("sensfunc") is None, "it runs now; the registry answers"


#: Tasks whose steps do not yet say what they do to the data. This started as a
#: backlog of seventeen and is now empty, so it has turned into a standard: a
#: new task that explains nothing fails this test until it does, or until
#: someone decides to put it here and say why.
UNEXPLAINED: set[str] = set()


def _example_params(cls):
    """Fill a task's required parameters with something plausible, or give up."""
    import typing

    values = {}
    for name, field in cls.Params.model_fields.items():
        if not field.is_required():
            continue
        annotation = field.annotation
        origin = typing.get_origin(annotation)
        if origin is list:
            values[name] = ["example.fits"]
        elif origin is typing.Literal:
            values[name] = typing.get_args(annotation)[0]
        elif annotation is int:
            values[name] = 1
        elif annotation is float:
            values[name] = 1.0
        else:
            values[name] = "example.fits"
    try:
        return cls.Params(**values)
    except Exception:
        return None


def test_the_unexplained_backlog_is_exactly_what_we_think_it_is():
    """Section 2.1 is the founding complaint: things happen behind the panels.

    A step that changes data and cannot say what it did is that complaint,
    unanswered. This does not demand every task explain itself today -- it
    demands that we know which ones do not.

    A task whose plan reads the actual files cannot be checked from here, so
    it is not counted either way; those have their own tests against real data
    (ccdproc's is in test_ccdred.py).
    """
    explained, unexplained, needs_data = set(), set(), set()
    for name, cls in list_tasks().items():
        params = _example_params(cls)
        if params is None:
            continue
        try:
            steps = cls().plan(params)
        except Exception:
            needs_data.add(name)
            continue
        (explained if any(step.explained() for step in steps)
         else unexplained).add(name)

    checked = unexplained | explained
    assert unexplained == UNEXPLAINED & checked, (
        f"the backlog moved.\n"
        f"  newly explained: {sorted(UNEXPLAINED & checked - unexplained)}\n"
        f"  newly unexplained: {sorted(unexplained - UNEXPLAINED)}\n"
        f"Every task says what its steps do to the data. Write the `why` of the "
        f"new one's steps, or add it to UNEXPLAINED here with a reason."
    )
    assert len(explained) > 15, "the check itself must still be reaching the tasks"
    assert "ccdproc" in needs_data, (
        "ccdproc's plan reads real headers; if that changed, check it here "
        "rather than leaving it uncounted"
    )


def test_an_iraf_name_that_harwood_runs_under_another_name_says_which():
    """`apedit` is part of `apall`. "Did you mean apall?" makes that look like a
    guess; the map knows, so it should say so."""
    answer = migration.explain("apedit")
    assert answer is not None
    assert "apall" in answer
    assert migration.explain("daofind") is None, \
        "a name harwood runs under its own name needs no answer"


def test_what_harwood_adds_is_listed_apart_from_what_iraf_had():
    """The IRAF side of the map must stay an inventory of IRAF.

    Someone deciding whether to migrate reads `tasks:` to learn what they
    lose. A harwood invention smuggled in there would make that count wrong,
    so additions live in their own section and are checked the same way.
    """
    from harwood.core.registry import exists

    extra = migration.added()
    assert extra, "there is at least one, and the section has to be readable"
    for name, entry in extra.items():
        assert name not in migration.tasks(), (
            f"'{name}' is an IRAF task after all; it belongs in tasks:")
        assert entry["status"] in ("available", "partial"), name
        assert entry.get("note"), f"'{name}' does not say what it is for"
        assert exists(name), f"the map promises '{name}' and it does not exist"


def test_the_counts_in_the_map_are_the_counts_in_the_map():
    """The header comment is documentation of the file it sits in, and
    documentation that is not checked drifts. It had: partial 10, when there
    were 21.
    """
    import re
    from collections import Counter
    from pathlib import Path

    from harwood.core.registry import list_tasks

    tasks = migration.tasks()
    header = Path(migration._DATA).read_text(encoding="utf-8")
    header = header[:header.index("meta:")]

    status = Counter(entry["status"] for entry in tasks.values())
    for name, count in status.items():
        assert re.search(rf"{name}: {count}\b", header), \
            f"the header does not say {name}: {count}"

    assert re.search(rf"\b{len(tasks)} distinct task names", header)

    milestone = Counter(entry["milestone"] for entry in tasks.values()
                        if entry.get("milestone"))
    running = list_tasks()
    done = sum(1 for name, entry in tasks.items()
               if entry.get("milestone") == "mvp"
               and entry.get("harwood", name) in running)
    assert re.search(rf"mvp: {milestone['mvp']} \({done} of them running\)", header)


def test_a_finished_task_keeps_the_milestone_it_was_planned_for():
    """Otherwise progress through the MVP cannot be counted: every completed
    task drops out of the denominator as well as the numerator, and the map
    can only say what is left."""
    from harwood.core.registry import list_tasks

    running = list_tasks()
    forgotten = [name for name, entry in migration.tasks().items()
                 if entry.get("harwood", name) in running
                 and not entry.get("milestone")]
    assert not forgotten, f"built but no longer counted: {', '.join(forgotten)}"
