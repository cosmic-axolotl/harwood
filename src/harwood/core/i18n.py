"""The languages harwood speaks on the screen: English, Portuguese and Spanish.

What is translated is what a person reads: the window's menus, buttons and
hints, the notices (`core.advice`: what happened, and what to do), the
explanations of a plan's steps (`--explain`), and the documentation. What is
never translated is what has to mean the same everywhere: task and parameter
names and the descriptions of the parameters, keys, what IRAF's own tasks print
(a log, :show, a prompt) and every file harwood writes -- logs, run.yaml,
HISTORY cards, databases stay in English, so that a reduction reads the same
whoever ran it.

The language is, in order: HARWOOD_LANG; the one chosen in the window (saved
with the other preferences, ~/.config/harwood/interface.json, "language"); the
system's (LC_ALL, LC_MESSAGES, LANG, LANGUAGE); English.

The catalogues are gettext's .po files, data/i18n/<language>.po, read here with
no compiling step. An entry is either a text, translated when it is met exactly,
or -- flagged python-brace-format -- a mould: a notice written with an f-string
("cursor file '{0}' does not exist") is matched with its slots filled by
anything, and the translation gets the same values in its own slots, in its own
order. So a notice needs nothing done where it is raised: it is translated where
it is shown. `python -m harwood.core.i18n extract` finds the texts in the code
and adds the new ones to the catalogues.
"""

from __future__ import annotations

import ast
import json
import locale
import os
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

LANGUAGES = ("en", "pt", "es")
#: Each language in its own words, as a menu offers it.
NAMES = {"en": "English", "pt": "Português", "es": "Español"}
CATALOGUES = Path(__file__).resolve().parent.parent / "data" / "i18n"
PACKAGE = Path(__file__).resolve().parent.parent

_chosen: str | None = None      # set_language's, for this process
#: The fewest letters a mould's fixed text needs to be used as one.
MOULD_LETTERS = 10


# ------------------------------------------------------------ the language

def _preferences_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "harwood" / "interface.json"


def saved_choice() -> str:
    """What the window saved: a language, or "system"."""
    try:
        stored = json.loads(_preferences_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "system"
    choice = stored.get("language", "system") if isinstance(stored, dict) else "system"
    return choice if choice in LANGUAGES else "system"


def save_choice(choice: str) -> Path:
    """Keep `choice` (a language, or "system") with the window's preferences, the
    other choices there left as they are."""
    path = _preferences_path()
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored = stored if isinstance(stored, dict) else {}
    except (OSError, ValueError):
        stored = {}
    stored["language"] = choice if choice in LANGUAGES else "system"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    return path


def _code(text: str | None) -> str | None:
    """'pt_BR.UTF-8', 'Portuguese_Brazil', 'es' -> a language of ours, or None."""
    if not text:
        return None
    word = text.strip().lower()
    for name, code in (("portuguese", "pt"), ("spanish", "es"), ("english", "en")):
        if word.startswith(name):
            return code
    code = re.split(r"[_.@:-]", word)[0]
    return code if code in LANGUAGES else None


def system_language() -> str:
    """The system's, from the locale's variables, then the locale itself."""
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(name)
        if value and value not in ("C", "POSIX", "C.UTF-8"):
            return _code(value) or "en"
    for part in os.environ.get("LANGUAGE", "").split(":"):
        if _code(part):
            return _code(part)
    try:
        found = _code(locale.getlocale()[0])
    except ValueError:
        found = None
    return found or "en"


def language() -> str:
    """The language now: HARWOOD_LANG, then the choice made here or saved, then the
    system's."""
    forced = _code(os.environ.get("HARWOOD_LANG"))
    if forced:
        return forced
    if _chosen:
        return _chosen
    choice = saved_choice()
    return choice if choice != "system" else system_language()


def set_language(code: str | None) -> None:
    """Speak `code` from now on in this process ("system" or None: as the system
    does). Saving the choice is the window's (its preferences)."""
    global _chosen
    _chosen = code if code in LANGUAGES else None


# ------------------------------------------------------------ catalogues

@dataclass
class Catalogue:
    exact: dict[str, str] = field(default_factory=dict)
    moulds: list[tuple[re.Pattern, str]] = field(default_factory=list)
    #: The first characters of a mould's text, for finding the ones worth trying.
    starts: dict[str, list[int]] = field(default_factory=dict)


def _unquote(text: str) -> str:
    return re.sub(r'\\(.)', lambda m: {"n": "\n", "t": "\t"}.get(m[1], m[1]), text)


def read_po(path: Path) -> list[dict]:
    """The entries of a .po file: msgid, msgstr, flags, comments and references."""
    entries, entry, key = [], None, None

    def close():
        # the header (msgid "") is the file's, written again by write_po
        if entry is not None and entry.get("msgid"):
            entries.append(entry)

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            close()
            entry, key = None, None
            continue
        if entry is None:
            entry = {"flags": set(), "comments": [], "refs": []}
        if line.startswith("#,"):
            entry["flags"] |= {f.strip() for f in line[2:].split(",")}
        elif line.startswith("#:"):
            entry["refs"].append(line[2:].strip())
        elif line.startswith("#"):
            entry["comments"].append(line)
        elif line.startswith(("msgid ", "msgstr ")):
            key, rest = line.split(" ", 1)
            entry[key] = _unquote(rest.strip()[1:-1])
        elif line.startswith('"') and key:
            entry[key] += _unquote(line[1:-1])
    close()
    return entries


def _mould(text: str) -> re.Pattern:
    """A mould's text as a pattern: {n} matches anything, {{ and }} are braces."""
    parts = re.split(r"(\{\d+\})", text)
    pattern = "".join(f"(?P<s{p[1:-1]}>.*?)" if re.fullmatch(r"\{\d+\}", p)
                      else re.escape(p.replace("{{", "{").replace("}}", "}"))
                      for p in parts)
    return re.compile(pattern, re.S)


@cache
def catalogue(code: str) -> Catalogue:
    found = Catalogue()
    path = CATALOGUES / f"{code}.po"
    if code == "en" or not path.exists():
        return found
    for entry in read_po(path):
        source, target = entry["msgid"], entry.get("msgstr", "")
        if not source or not target or "fuzzy" in entry["flags"]:
            continue
        if "python-brace-format" in entry["flags"] and re.search(r"\{\d+\}", source):
            # a mould of little fixed text -- "{0} and {1}", "use {0}" -- would catch
            # any text with those words in it, and translate half of it: left out
            if len(re.sub(r"\{\d+\}|[^A-Za-z]", "", source)) < MOULD_LETTERS:
                continue
            found.starts.setdefault(_lead(source), []).append(len(found.moulds))
            found.moulds.append((_mould(source), target))
        else:
            found.exact[source] = target
    return found


def _lead(text: str) -> str:
    """The literal text a mould starts with, up to 6 characters ('' if a slot)."""
    head = re.split(r"\{\d+\}", text, maxsplit=1)[0]
    return head.replace("{{", "{").replace("}}", "}")[:6]


def tr(text: str, code: str | None = None) -> str:
    """`text` in the language (now's, or `code`): exactly, or by a mould; as it is
    when there is no translation."""
    if not text:
        return text
    code = code or language()
    if code == "en":
        return text
    found = catalogue(code)
    if text in found.exact:
        return found.exact[text]
    candidates = found.starts.get(text[:6], []) + [
        i for lead in {text[:n] for n in range(6)} for i in found.starts.get(lead, [])]
    for index in candidates:
        pattern, target = found.moulds[index]
        match = pattern.fullmatch(text)
        if match:
            return _fill(target, match.groupdict())
    return text


def _fill(target: str, values: dict) -> str:
    return re.sub(r"\{(\d+)\}", lambda m: values.get(f"s{m[1]}", m[0]), target)


def N_(text: str) -> str:
    """A text marked to be translated where it is shown, left as it is here."""
    return text


def trf(text: str, *values, code: str | None = None) -> str:
    """A text with slots, {0} {1}..., translated and then filled."""
    out = tr(text, code)
    return re.sub(r"\{(\d+)\}", lambda m: str(values[int(m[1])])
                  if int(m[1]) < len(values) else m[0], out)


# ------------------------------------------------------------- extraction

#: Calls whose first two arguments are a notice and its suggestion.
NOTICES = {("advice", "fail"), ("advice", "warn"), ("advice", "note"),
           ("advice", "problem"), (None, "HarwoodError"), (None, "Advice")}
#: Calls whose first argument is a text of the window's (viewer.lang).
MARKED = {"tr", "trf", "bind", "action", "N_", "_t", "_tf"}


def _piece(node: ast.AST, slots: list) -> str | None:
    """A text as a mould: literal parts kept (braces doubled), anything else a slot."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.replace("{", "{{").replace("}", "}}")
    if isinstance(node, ast.JoinedStr):
        out = ""
        for value in node.values:
            if isinstance(value, ast.Constant):
                out += str(value.value).replace("{", "{{").replace("}", "}}")
            else:
                out += f"{{{len(slots)}}}"
                slots.append(value)
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _piece(node.left, slots), _piece(node.right, slots)
        if left is None and right is None:
            return None
        if left is None:
            left = f"{{{len(slots)}}}"
            slots.append(node.left)
        if right is None:
            right = f"{{{len(slots)}}}"
            slots.append(node.right)
        return left + right
    return None


def _texts(node: ast.AST) -> list[tuple[str, bool]]:
    """The texts an argument can be: (text, is a mould)."""
    if isinstance(node, ast.IfExp):
        return _texts(node.body) + _texts(node.orelse)
    slots: list = []
    text = _piece(node, slots)
    if text is None or not text.strip():
        return []
    if not slots:
        return [(text.replace("{{", "{").replace("}}", "}"), False)]
    return [(text, True)]


def _name(func: ast.AST) -> tuple[str | None, str]:
    if isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else None
        return owner, func.attr
    if isinstance(func, ast.Name):
        return None, func.id
    return None, ""


def extract(root: Path = PACKAGE) -> dict[tuple[str, bool], list[str]]:
    """Every translatable text of the package: notices and suggestions, the `why`
    of plan steps (Step(..., why=...) and the *WHY dictionaries), task
    descriptions, the keys' descriptions (*KEYS, *HELP, WINDOW, MOUSE) and the
    window's marked texts (tr, trf, bind, action, N_). Text -> where."""
    found: dict[tuple[str, bool], list[str]] = {}

    def add(node, where):
        for text, mould in _texts(node):
            found.setdefault((text, mould), []).append(where)

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or path.parts[-2:] == ("core", "i18n.py"):
            continue
        rel = path.relative_to(root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            where = f"{rel}:{getattr(node, 'lineno', 0)}"
            if isinstance(node, ast.Call):
                owner, name = _name(node.func)
                if (owner, name) in NOTICES or (None, name) in NOTICES:
                    for arg in node.args[:2]:
                        add(arg, where)
                    for kw in node.keywords:
                        if kw.arg in ("message", "fix"):
                            add(kw.value, where)
                elif name in MARKED and owner in (None, "lang", "i18n") and node.args:
                    arg = node.args[2] if name == "bind" and len(node.args) > 2 else \
                        node.args[1] if name == "action" and len(node.args) > 1 else \
                        node.args[0]
                    add(arg, where)
                elif name in ("_make_step", "_make_preset") and len(node.args) > 1:
                    add(node.args[1], where)        # a preset's description
                elif name == "Step":
                    for kw in node.keywords:
                        if kw.arg == "why":
                            add(kw.value, where)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                value = node.value
                if any(n.endswith("WHY") for n in names) and isinstance(value, ast.Dict):
                    for item in value.values:
                        add(item, where)
                elif any(n.endswith(("KEYS", "HELP")) or n in ("WINDOW", "MOUSE")
                         for n in names):
                    # a panel's keys: {key: what} or [(key, what), ...]
                    if isinstance(value, ast.Dict):
                        for item in value.values:
                            add(item, where)
                    elif isinstance(value, (ast.List, ast.Tuple)):
                        for item in value.elts:
                            if isinstance(item, ast.Tuple) and len(item.elts) >= 2:
                                add(item.elts[1], where)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == "description"
                            for t in item.targets):
                        add(item.value, f"{rel}:{item.lineno}")
    # the package tree's one or two lines on each package, and the sections of the
    # documentation's contents
    import yaml

    nav = yaml.safe_load((root / "docsite" / "pages" / "nav.yaml").read_text(encoding="utf-8"))
    for section in nav or []:
        found.setdefault((section["section"], False), []).append(
            "harwood/docsite/pages/nav.yaml")

    tree = yaml.safe_load((root / "data" / "packages.yaml").read_text(encoding="utf-8"))
    for path, info in (tree or {}).items():
        if isinstance(info, dict) and info.get("brief"):
            found.setdefault((" ".join(str(info["brief"]).split()), False), []).append(
                f"harwood/data/packages.yaml:{path or '(top)'}")
    return found


def _quote(text: str) -> str:
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t")
    if "\n" not in text:
        return f'"{text}"'
    lines = text.split("\n")
    parts = [line + "\\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return '""\n' + "\n".join(f'"{p}"' for p in parts)


def write_po(path: Path, entries: list[dict]) -> None:
    out = ['# harwood\'s texts, translated. Made by "python -m harwood.core.i18n '
           'extract";', "# the translations are kept when it runs again.", 'msgid ""',
           'msgstr ""', '"Content-Type: text/plain; charset=UTF-8\\n"', ""]
    for entry in entries:
        out += entry.get("comments", [])
        out += [f"#: {ref}" for ref in entry.get("refs", [])[:3]]
        if entry.get("flags"):
            out.append("#, " + ", ".join(sorted(entry["flags"])))
        out.append(f"msgid {_quote(entry['msgid'])}")
        out.append(f"msgstr {_quote(entry.get('msgstr', ''))}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def update(code: str, root: Path = PACKAGE) -> tuple[int, int]:
    """The catalogue of `code` with every text there is now: translations kept,
    new texts added empty, texts no longer in the code dropped. (texts, untranslated)"""
    path = CATALOGUES / f"{code}.po"
    old = {e["msgid"]: e for e in read_po(path)} if path.exists() else {}
    entries = []
    for (text, mould), refs in sorted(extract(root).items(), key=lambda kv: kv[1][0]):
        before = old.get(text, {})
        flags = {"python-brace-format"} if mould else set()
        flags |= {f for f in before.get("flags", set()) if f == "fuzzy"}
        entries.append({"msgid": text, "msgstr": before.get("msgstr", ""),
                        "flags": flags, "refs": refs,
                        "comments": [c for c in before.get("comments", [])
                                     if c.startswith("# ")]})
    CATALOGUES.mkdir(parents=True, exist_ok=True)
    write_po(path, entries)
    catalogue.cache_clear()
    return len(entries), sum(1 for e in entries if not e["msgstr"])


def merge(path: Path) -> dict[str, int]:
    """Translations from a JSON file, {english: [portuguese, spanish]}, into the
    catalogues, over what they had. How many texts each got."""
    given = json.loads(Path(path).read_text(encoding="utf-8"))
    done = {}
    for index, code in enumerate(LANGUAGES[1:]):
        target = CATALOGUES / f"{code}.po"
        entries = read_po(target)
        count = 0
        for entry in entries:
            if entry["msgid"] in given and given[entry["msgid"]][index]:
                entry["msgstr"] = given[entry["msgid"]][index]
                entry["flags"].discard("fuzzy")
                count += 1
        write_po(target, entries)
        done[code] = count
    catalogue.cache_clear()
    return done


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["extract"]:
        for code in LANGUAGES[1:]:
            total, missing = update(code)
            print(f"{code}: {total} texts, {missing} untranslated")
    elif sys.argv[1:2] == ["merge"]:
        for name in sys.argv[2:]:
            print(name, merge(Path(name)))
