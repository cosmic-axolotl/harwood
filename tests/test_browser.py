"""The click-only tab: IRAF's package tree, and a form for each task.

The tree is held to what an IRAF user expects (daofind is under noao >
digiphot > apphot) and to what was promised about it: every level says in a
line or two what is in it, and there is always a visible way back. The form
is held to the rule that makes it trustworthy -- what it runs is exactly the
command it shows, and that command is what the command line would accept.
"""

import numpy as np
import pytest

from harwood.core import packages
from harwood.core.registry import get_class, list_tasks

# --------------------------------------------------------------- the tree


def test_the_tree_is_irafs():
    root = packages.tree()
    assert packages.locate("daofind", root) == ["noao.digiphot.apphot"]
    assert packages.locate("ccdproc", root) == ["noao.imred.ccdred"]
    assert packages.locate("imcombine", root) == ["images.immatch"]
    assert packages.locate("observatory", root) == ["noao"]
    assert packages.find("noao.digiphot", root).parent_path == "noao"
    assert packages.find("noao", root).parent_path == ""
    assert packages.find("", root).parent_path is None


def test_every_harwood_task_has_a_place_in_it():
    root = packages.tree()
    homeless = [name for name in list_tasks() if not packages.locate(name, root)]
    assert homeless == []


def test_every_package_says_briefly_what_is_in_it():
    briefs = []

    def walk(node):
        briefs.append((node.path, node.brief))
        for child in node.children:
            walk(child)

    walk(packages.tree())
    assert all(brief for _, brief in briefs), [p for p, b in briefs if not b]
    # Brief means brief: the long explanation is the documentation's.
    assert max(len(brief) for _, brief in briefs) <= 130


def test_a_wrong_path_says_what_is_there():
    with pytest.raises(KeyError, match="digiphot"):
        packages.find("noao.digifot")


def test_short_iraf_paths_are_made_whole():
    assert packages.full_path("digiphot.apphot") == "noao.digiphot.apphot"
    assert packages.full_path("images.imutil") == "images.imutil"
    assert packages.full_path("tv") == "images.tv"


# ------------------------------------------- the command the form will show


def test_a_second_list_is_written_the_way_the_command_line_reads_it():
    """The first list among the positionals takes a word per item; a later
    one has to be joined by commas or the first swallows it."""
    import click

    from harwood.cli import build
    from harwood.cli.main import app
    from harwood.core import replay

    values = {"oldnames": ["a.fits", "b.fits"], "newnames": ["c.fits", "d.fits"]}
    model = get_class("imrename").Params
    line = replay.command("imrename", model, values)
    assert line == "harwood imrename a.fits b.fits c.fits,d.fits"
    command = app.get_command(click.Context(app), "imrename")
    parsed = command.make_context("imrename", line.split()[2:]).params
    assert build.collect(model, parsed) == values


# ---------------------------------------------------------------- the GUI


@pytest.fixture
def browser():
    pytest.importorskip("PySide6")
    from harwood.viewer._qt import application
    from harwood.viewer.browser import TaskBrowser

    application()
    sent = []

    def runner(line):
        sent.append(line)
        return True, "ran\n"

    widget = TaskBrowser(runner)
    widget.sent = sent
    yield widget
    widget.close()


def _cards(widget):
    from harwood.viewer.browser import Card

    return [widget.cards.itemAt(i).widget() for i in range(widget.cards.count())
            if isinstance(widget.cards.itemAt(i).widget(), Card)]


def test_going_in_and_coming_back(browser):
    assert not browser.back_button.isEnabled(), "nowhere to go back to from the top"
    names = [card.name for card in _cards(browser)]
    assert "noao" in names and "images" in names
    assert "softools" not in names, "a package harwood has nothing in stays out of the way"

    next(c for c in _cards(browser) if c.name == "noao").click()
    next(c for c in _cards(browser) if c.name == "digiphot").click()
    assert browser.node.path == "noao.digiphot"
    assert browser.back_button.text().endswith("Back to noao")
    assert 'href="noao"' in browser.crumbs.text(), "every part of the path is a link"
    assert browser.brief.text() == packages.find("noao.digiphot").brief

    browser.back_button.click()
    assert browser.node.path == "noao"
    browser.crumbs.linkActivated.emit("")
    assert browser.node.path == ""


def test_a_task_goes_back_to_its_package(browser):
    browser.go("noao.digiphot.apphot")
    next(c for c in _cards(browser) if c.name == "daofind").click()
    assert browser.task == "daofind" and browser.pages.currentIndex() == 1
    assert browser.back_button.text().endswith("Back to apphot")
    assert browser.crumbs.text().endswith("<b>daofind</b>")
    browser.back()
    assert browser.task is None and browser.node.path == "noao.digiphot.apphot"


def test_the_missing_tasks_are_there_when_asked_for(browser):
    browser.go("images.imutil")
    shown = {card.name for card in _cards(browser)}
    browser.missing_box.setChecked(True)
    everything = {card.name for card in _cards(browser)}
    assert shown < everything
    assert "listpixels" in everything


def test_searching_finds_a_task_wherever_it_lives(browser):
    browser.search.setText("daofi")
    assert [card.name for card in _cards(browser)] == ["daofind"]
    browser.search.returnPressed.emit()
    assert browser.task == "daofind" and browser.node.path == "noao.digiphot.apphot"


def test_the_form_runs_the_command_it_shows(browser):
    from harwood.core import replay

    browser.open_task("imarith")
    assert not browser.run_button.isEnabled(), "nothing to work on yet"
    assert "operand1 is needed" in browser.problems.text()
    fields = {f.name: f for f in browser.fields}
    fields["operand1"].editor.setText("a.fits")
    fields["op"].editor.setCurrentText("*")
    fields["operand2"].editor.setText("3")
    fields["result"].editor.setText("b.fits")
    fields["divzero"].editor.setText("1")
    assert browser.run_button.isEnabled()

    expected = replay.command("imarith", get_class("imarith").Params, {
        "operand1": "a.fits", "op": "*", "operand2": "3", "result": "b.fits",
        "divzero": 1.0})
    assert browser.command.text() == expected == \
        "harwood imarith a.fits '*' 3 b.fits --divzero 1"
    browser.plan_button.click()
    browser.run_button.click()
    assert browser.sent == [expected + " --plan", expected]
    assert "done" in browser.outcome.toPlainText()


def test_a_bad_value_is_said_in_words_and_stops_the_run(browser):
    browser.open_task("daofind")
    fields = {f.name: f for f in browser.fields}
    fields["image"].editor.setText("obj.fits")
    fields["threshold"].editor.setText("four")
    assert not browser.run_button.isEnabled()
    assert "threshold" in browser.problems.text()
    assert fields["threshold"].editor.property("invalid") == "true"
    fields["threshold"].editor.setText("5")
    assert browser.run_button.isEnabled()
    assert "--threshold 5" in browser.command.text()


def test_indef_and_lists_are_written_as_iraf_writes_them(browser):
    browser.open_task("imstatistics")
    fields = {f.name: f for f in browser.fields}
    fields["input"].set_files(["a.fits", "b.fits"])
    fields["lower"].editor.setText("INDEF")
    fields["upper"].editor.setText("30000")
    assert browser.command.text() == "harwood imstatistics a.fits b.fits --upper 30000"


def test_a_form_starts_with_the_image_on_display(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from harwood.viewer._qt import application
    from harwood.viewer.browser import TaskBrowser

    application()
    widget = TaskBrowser(lambda line: (True, ""), current_image=lambda: "obj.fits")
    widget.open_task("daofind")
    assert widget.command.text() == "harwood daofind obj.fits"
    widget.close()


# ------------------------------------------------------- inside the window


@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from harwood.viewer import theme
    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.chdir(tmp_path)
    application()
    w = HarwoodWindow()
    yield w
    w.close()
    theme.forget(w._restyle)


def test_a_click_runs_through_the_same_command_line_as_typing(window, tmp_path):
    from astropy.io import fits

    fits.writeto(tmp_path / "a.fits", np.arange(16, dtype=np.float32).reshape(4, 4))
    browser = window.browser
    browser.open_task("imstatistics")
    next(f for f in browser.fields if f.name == "input").editor.setText("a.fits")
    ok, printed = browser.run()
    assert ok, printed
    session = window.session.toPlainText()
    assert "hw> imstatistics a.fits" in session, "shown as if it had been typed"
    assert "7.5" in printed, "the mean of 0..15"

    ok, printed = window.run_line("harwood imstatistics missing.fits")
    assert not ok


def test_edit_as_text_puts_the_command_at_the_prompt(window):
    window.browser.open_task("daofind")
    window.browser.to_prompt.emit(window.browser.command.text())
    assert window.prompt.text() == "daofind"
