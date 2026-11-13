"""harwood in English, Portuguese and Spanish (core/i18n.py).

The language: HARWOOD_LANG, then what was chosen and saved, then the system's.
The catalogues: every text the code has is in them (the extraction is up to
date), a translation keeps the slots of its English, and everything the window,
the terminal's own words and the documentation's frame say is translated. A notice
made with an f-string is translated by its mould where it is shown. What is never
translated -- the parameters' descriptions, IRAF's output -- stays English. The
window changes language while open; the documentation is built in all three,
each page linking to itself in the others."""

from __future__ import annotations

import json
import re

import pytest

from harwood.core import i18n

WINDOW = ("harwood/viewer/", "harwood/cli/", "harwood/docsite/",
          "harwood/data/packages.yaml")


@pytest.fixture
def no_forcing(monkeypatch):
    monkeypatch.delenv("HARWOOD_LANG", raising=False)
    for name in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    i18n.set_language(None)
    yield
    i18n.set_language(None)


# ------------------------------------------------------------ the language

@pytest.mark.parametrize("value, code", [("pt_BR.UTF-8", "pt"), ("es_AR.UTF-8", "es"),
                                         ("en_GB.UTF-8", "en"), ("de_DE.UTF-8", "en"),
                                         ("Portuguese_Brazil.1252", "pt"), ("es", "es")])
def test_the_system_s_language_is_read_from_the_locale(no_forcing, monkeypatch, value, code):
    monkeypatch.setenv("LANG", value)
    assert i18n.system_language() == code
    assert i18n.language() == code


def test_what_is_saved_wins_over_the_system_and_harwood_lang_over_both(no_forcing,
                                                                     monkeypatch):
    monkeypatch.setenv("LANG", "pt_BR.UTF-8")
    path = i18n.save_choice("es")
    assert json.loads(path.read_text())["language"] == "es"
    assert i18n.language() == "es"
    monkeypatch.setenv("HARWOOD_LANG", "en")
    assert i18n.language() == "en"
    monkeypatch.delenv("HARWOOD_LANG")
    i18n.save_choice("system")
    assert i18n.language() == "pt"


def test_saving_the_language_keeps_the_other_preferences(no_forcing):
    path = i18n._preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"theme": "light", "scale": 1.2}))
    i18n.save_choice("pt")
    assert json.loads(path.read_text()) == {"theme": "light", "scale": 1.2,
                                            "language": "pt"}


# -------------------------------------------------------------- texts

def test_a_text_is_translated_exactly_and_a_notice_by_its_mould():
    assert i18n.tr("Copy", "pt") == "Copiar"
    assert i18n.tr("Copy", "es") == "Copiar"
    assert i18n.tr("Copy", "en") == "Copy"
    # cli/line.py raises f"nothing before '{after}'": the mould fills its slot
    assert i18n.tr("nothing before '&&'", "pt") == "nada antes de '&&'"
    assert i18n.tr("nothing after 'a b; c'", "es") == "nada después de 'a b; c'"
    assert i18n.trf("{0} of {1} tasks in harwood", 3, 9, code="pt") == \
        "3 de 9 tarefas no harwood"
    assert i18n.tr("not a text harwood has", "pt") == "not a text harwood has"


def test_the_catalogues_hold_every_text_the_code_has():
    """`python -m harwood.core.i18n extract` after a text is added or changed."""
    texts = {text for text, _ in i18n.extract()}
    for code in i18n.LANGUAGES[1:]:
        held = {e["msgid"] for e in i18n.read_po(i18n.CATALOGUES / f"{code}.po")}
        assert texts == held, f"{code}.po is behind the code: run the extraction"


@pytest.mark.parametrize("code", ["pt", "es"])
def test_a_translation_keeps_the_slots_of_its_english(code):
    for entry in i18n.read_po(i18n.CATALOGUES / f"{code}.po"):
        if entry.get("msgstr"):
            slots = sorted(re.findall(r"\{\d+\}", entry["msgid"]))
            assert sorted(re.findall(r"\{\d+\}", entry["msgstr"])) == slots, entry["msgid"]


@pytest.mark.parametrize("code", ["pt", "es"])
def test_everything_the_window_and_the_documentation_s_frame_say_is_translated(code):
    missing = [e["msgid"] for e in i18n.read_po(i18n.CATALOGUES / f"{code}.po")
               if not e.get("msgstr") and any(r.startswith(WINDOW) for r in e["refs"])]
    assert missing == []


def test_the_terminal_says_its_notices_in_the_language(monkeypatch):
    from rich.console import Console

    from harwood.cli import main
    from harwood.core.advice import Advice

    monkeypatch.setenv("HARWOOD_LANG", "pt")
    out = Console(record=True, width=100)
    main._print_advice([Advice("harwood does not pipe one command into the next",
                               "write '&&' or ';' between commands; to filter what a task "
                               "prints, run it from the shell: harwood ... | grep ...")],
                       where=out)
    said = out.export_text()
    assert "aviso" in said and "não encadeia" in said and "tente:" in said


def test_parameter_descriptions_stay_in_english(monkeypatch):
    from click.testing import CliRunner

    from harwood.cli.main import app

    monkeypatch.setenv("HARWOOD_LANG", "pt")
    printed = CliRunner().invoke(app, ["fitcoords", "--help"]).output
    assert "Order of the surface in x" in printed


# ------------------------------------------------------- the window

def test_the_window_speaks_another_language_without_being_made_again(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer import theme
    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.delenv("HARWOOD_LANG")
    monkeypatch.chdir(tmp_path)
    application()
    i18n.set_language("en")
    window = HarwoodWindow()
    try:
        titles = [m.title() for m in window.menuBar().findChildren(type(
            window.menuBar().actions()[0].menu())) if m.title()]
        assert "&File" in titles
        window.set_language("pt")
        assert theme.Preferences.load().language == "pt"
        titles = [a.text() for a in window.menuBar().actions()]
        assert titles[:2] == ["&Arquivo", "E&xibir"]
        assert window.tasks_dock.windowTitle() == "Tarefas"
        assert window.browser.run_button.text() == "▶  Executar"
        assert window.key_guide.title.startswith("Exiba uma imagem")
        assert window._language_actions["pt"].isChecked()
        window.set_language("es")
        assert window.tasks_dock.windowTitle() == "Tareas"
        assert window.browser.brief.text().startswith("Todas las tareas")
        window.set_language("en")
        assert window.tasks_dock.windowTitle() == "Tasks"
    finally:
        window.close()
        theme.forget(window._restyle)
        i18n.set_language(None)


# ------------------------------------------------- the documentation

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from harwood import docsite

    return docsite.build(tmp_path_factory.mktemp("site"))


def test_the_documentation_is_built_in_three_languages(built):
    for code, root in (("en", built), ("pt", built / "pt"), ("es", built / "es")):
        page = (root / "tasks" / "fitcoords.html").read_text(encoding="utf-8")
        assert f'<html lang="{code}"' in page
        assert (root / "search-index.js").exists()
    assert "Parâmetros" in (built / "pt" / "tasks" / "fitcoords.html").read_text()
    assert "Parámetros" in (built / "es" / "tasks" / "fitcoords.html").read_text()
    # the parameters' own descriptions are English in every language
    assert "Order of the surface in x" in (built / "pt" / "tasks" / "fitcoords.html"
                                           ).read_text()


def test_each_page_links_to_itself_in_the_other_languages(built):
    page = (built / "pt" / "start" / "install.html").read_text(encoding="utf-8")
    links = dict(re.findall(r'href="([^"]+)" hreflang="(\w+)"', page))
    assert sorted(links.values()) == ["en", "es", "pt"]
    for href in links:
        assert ((built / "pt" / "start") / href).resolve().exists(), href
    assert 'hreflang="pt" lang="pt" title="Português" aria-current="true"' in page


def test_a_page_not_translated_says_so_in_the_language(built):
    from harwood.docsite import builder

    for code in ("pt", "es"):
        for name in ("concepts/outputs.md", "tasks/fitcoords.md"):
            page = (built / code / name.replace(".md", ".html")).read_text()
            translated = builder.translation(name, code).exists()
            said = i18n.tr("This page has not been translated yet; it is shown in English.",
                           code)
            assert (said in page) != translated, (code, name)


def test_a_translation_is_of_the_page_as_it_is_now():
    """A translation records the English it was made from; when the English changes,
    the translation is behind, and this says which."""
    from harwood.docsite import builder

    behind = []
    for path in sorted(builder.TRANSLATIONS.rglob("*.md")):
        code, name = path.relative_to(builder.TRANSLATIONS).parts[0], \
            "/".join(path.relative_to(builder.TRANSLATIONS).parts[1:])
        first = path.read_text(encoding="utf-8").splitlines()[0]
        made = re.match(r"<!-- from: (\w+) -->", first)
        assert made, f"{code}/{name}: no '<!-- from: ... -->' line"
        if made[1] != builder.source_stamp(name):
            behind.append(f"{code}/{name}")
    assert behind == []


@pytest.mark.parametrize("code", ["pt", "es"])
def test_every_task_s_one_line_and_every_key_is_translated(code):
    """What the Tasks panel and the Keys strip show, in the window's language."""
    from harwood.core.registry import list_tasks
    from harwood.viewer.keyguide import MOUSE, WINDOW

    held = {e["msgid"]: e.get("msgstr") for e in
            i18n.read_po(i18n.CATALOGUES / f"{code}.po")}
    missing = [cls.description for cls in list_tasks().values()
               if cls.description and not held.get(cls.description)]
    missing += [what for _, what in WINDOW + MOUSE if not held.get(what)]
    from harwood.core import examine

    missing += [what for what in examine.KEYS.values() if not held.get(what)]
    assert missing == []
