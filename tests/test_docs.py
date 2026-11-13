"""The documentation: it builds, it links, and every example on it is true.

The rule the recipes set (recipes/README.md) holds for every page: a command
on a page has been run, and the numbers quoted are the ones it produced. So
each example is run here on the sample night, from the state it says it
starts from, and checked for what the page says it prints; its Python tab is
run too, in a directory of its own, and has to write the same files. A link
to nothing, a task without a page, or a page without a manual fails here
rather than being found by a reader.
"""

import contextlib
import io
import os
import re
import shlex
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from harwood import docsite
from harwood.cli.main import app
from harwood.core.registry import list_tasks
from harwood.docsite import builder
from harwood.docsite import examples as ex


@pytest.fixture(scope="module")
def site():
    site = builder.collect()
    for page in site.pages:
        builder.render(page, site)
    return site


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return docsite.build(tmp_path_factory.mktemp("site"))


def test_the_site_builds_without_a_problem(site):
    assert site.problems == []


def test_every_task_has_a_page_with_every_parameter_and_a_manual(built):
    for name, cls in list_tasks().items():
        page = (built / "tasks" / f"{name}.html").read_text(encoding="utf-8")
        for field in cls.Params.model_fields:
            assert f'id="param-{field}"' in page, f"{name}: {field} missing"
        assert (builder.PAGES / "tasks" / f"{name}.md").exists(), f"{name}: no manual"


def test_every_link_inside_the_site_goes_somewhere(built):
    broken = []
    for page in built.rglob("*.html"):
        for target in re.findall(r'href="([^"#:]+)(?:#[^"]*)?"', page.read_text(
                encoding="utf-8")):
            if not (page.parent / target).resolve().exists():
                broken.append(f"{page.relative_to(built)} -> {target}")
    assert broken == []


def test_the_search_knows_every_page(built, site):
    index = (built / "search-index.js").read_text(encoding="utf-8")
    for page in site.pages:
        assert f'"u": "{page.url}"' in index


def test_the_pages_look_like_the_window_in_every_mode(built):
    from harwood.viewer import theme

    css = (built / "assets" / "look.css").read_text(encoding="utf-8")
    for palette in (theme.DARK, theme.LIGHT, theme.DARK_CB, theme.LIGHT_CB):
        assert f"--ground: {palette.ground};" in css and f"--accent: {palette.accent};" in css
    assert theme.CREAM in css, "the dyslexia mode's cream ground"


def test_a_broken_link_stops_the_build(monkeypatch, tmp_path):
    pages = tmp_path / "pages"
    shutil.copytree(builder.PAGES, pages)
    (pages / "index.md").write_text("# harwood\n\nSee [this](task:nosuchtask).\n",
                                    encoding="utf-8")
    monkeypatch.setattr(builder, "PAGES", pages)
    with pytest.raises(docsite.BuildError) as raised:
        docsite.build(tmp_path / "out")
    assert "task:nosuchtask" in str(raised.value)


def test_an_example_harwood_would_refuse_stops_the_build(monkeypatch, tmp_path):
    pages = tmp_path / "pages"
    shutil.copytree(builder.PAGES, pages)
    (pages / "index.md").write_text(
        "# harwood\n\n```example\nrun: harwood daofind obj001.fits --fwhm 3\n```\n",
                encoding="utf-8")
    monkeypatch.setattr(builder, "PAGES", pages)
    with pytest.raises(docsite.BuildError) as raised:
        docsite.build(tmp_path / "out")
    assert "--fwhm" in str(raised.value)


def test_the_three_tabs_say_the_same_thing():
    line = "harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3 --threshold 5"
    python = ex.python([line])
    assert "harwood.run(" in python and "fwhmpsf=3.0" in python and "threshold=5.0" in python
    steps = ex.clicks([line])[0]
    assert "noao › digiphot › apphot" in steps[0]
    assert "image: red/obj001.fits" in steps and "output: obj001.coo" in steps
    assert "fwhmpsf: 3.0" in steps[-2] and "threshold: 5.0" in steps[-2]
    assert ex.python(["harwood list"]) is None, "list has nothing to call"


# ------------------------------------------------------------ every example


@pytest.fixture(scope="module")
def stages(tmp_path_factory):
    """Each state of the sample night, made once and copied for each example."""
    made = {}

    def get(name):
        if name not in made:
            directory = tmp_path_factory.mktemp(f"stage-{name}")
            ex.prepare(name, directory)
            made[name] = directory
        return made[name]

    return get


def _all_examples():
    site = builder.collect()
    for page in site.pages:
        builder.render(page, site)
    return site.examples


EXAMPLES = _all_examples()


def _run(lines, where):
    previous = Path.cwd()
    os.chdir(where)
    try:
        printed, codes = "", []
        for line in lines:
            result = CliRunner().invoke(app, shlex.split(line)[1:])
            printed += result.output
            codes.append(result.exit_code)
        return printed, codes
    finally:
        os.chdir(previous)


def _squeeze(text):
    return " ".join(text.split())


def _slashes(text: str) -> str:
    """The pages write paths as red/obj001.fits, and Windows prints red\\obj001.fits."""
    return re.sub(r"(?<=[\w.])\\(?=[\w.])", "/", text) if os.sep == "\\" else text


@pytest.mark.parametrize("example", EXAMPLES,
                         ids=[f"{e.page}:{e.title or e.lines[0]}" for e in EXAMPLES])
def test_every_example_does_what_its_page_says(example, stages, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "100")
    work = tmp_path / "terminal"
    shutil.copytree(stages(example.after), work)
    printed, codes = _run(example.lines, work)
    if example.fails:
        assert all(code == 0 for code in codes[:-1]) and codes[-1] != 0, printed
    else:
        assert all(code == 0 for code in codes), printed
    for line in example.output.splitlines():
        assert _squeeze(line) in _squeeze(_slashes(printed)), \
            f"the page says it prints {line!r}"


@pytest.mark.parametrize("example", [e for e in EXAMPLES
                                     if not e.fails and ex.python(e.lines)],
                         ids=lambda e: f"{e.page}:{e.title or e.lines[0]}")
def test_every_python_tab_writes_what_the_terminal_writes(example, stages, tmp_path):
    def run_python(where):
        code = ex.python(example.lines)
        previous = Path.cwd()
        os.chdir(where)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, example.page, "exec"), {})
        finally:
            os.chdir(previous)

    def files(where):
        # The logs and the dated run directories differ by run, not by tab.
        return {p.relative_to(where) for p in where.rglob("*") if p.is_file()
                and not p.name.startswith("harwood.log") and "output" not in p.parts
                and ".state" not in p.parts}

    terminal, python = tmp_path / "terminal", tmp_path / "python"
    shutil.copytree(stages(example.after), terminal)
    shutil.copytree(stages(example.after), python)
    _run(example.lines, terminal)
    run_python(python)
    assert files(python) == files(terminal)


# ---------------------------------------------------------------- commands


def test_docs_builds_the_site_and_writes_the_sample(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["docs", "--build", str(tmp_path / "site")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "site" / "tasks" / "daofind.html").exists()
    result = runner.invoke(app, ["docs", "--sample", str(tmp_path / "night")])
    assert result.exit_code == 0 and (tmp_path / "night" / "obj001.fits").exists()


def test_docs_says_what_it_does_not_have(tmp_path):
    result = CliRunner().invoke(app, ["docs", "nosuchthing"])
    assert result.exit_code == 2 and "Nothing in the documentation" in result.output


def test_a_topic_finds_its_page():
    assert docsite.url_for("daofind") == "tasks/daofind.html"
    assert docsite.url_for("noao.digiphot.apphot") == "packages/noao.digiphot.apphot.html"
    assert docsite.url_for("concepts/outputs") == "concepts/outputs.html"
    assert docsite.url_for("") == "index.html"


def test_help_prints_the_same_manual_as_the_page():
    result = CliRunner().invoke(app, ["help", "imcopy"])
    assert "Example: Cut out a corner" in result.output
    assert "$ harwood imcopy" in result.output
    assert "harwood docs imcopy" in result.output


def test_the_built_site_is_rebuilt_when_a_page_changes(tmp_path, monkeypatch):
    out = tmp_path / "site"
    docsite.ensure(out)
    stamp = (out / ".source").read_text(encoding="utf-8")
    monkeypatch.setattr(builder, "_fingerprint", lambda: "something changed")
    docsite.ensure(out)
    assert (out / ".source").read_text(encoding="utf-8") == "something changed" != stamp


# ------------------------------------------------------------------ window


def test_the_window_passes_its_look_to_the_pages():
    pytest.importorskip("PySide6")
    from harwood.viewer import theme
    from harwood.viewer.docs import look_query

    theme.update(save=False, theme="light", colorblind=True, dyslexia=True)
    query = look_query()
    assert "look=light,colorblind,dyslexia" in query and "font=atkinson" in query


def test_f1_and_the_manual_button_open_the_page(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from harwood.viewer import _qt
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    opened = []
    _qt.application()
    window = HarwoodWindow()
    try:
        monkeypatch.setattr(window.docs, "show_page", opened.append)
        window.browser.open_task("daofind")
        window.browser.manual.emit("daofind")
        assert opened == ["tasks/daofind.html"]
        window.run_line("docs concepts/outputs")
        assert opened[-1] == "concepts/outputs.html"
    finally:
        _qt.set_window(None)
        window.close()
