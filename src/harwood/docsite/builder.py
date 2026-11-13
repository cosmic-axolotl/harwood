"""Building the documentation: one set of pages, for the window and the browser.

The window shows these files and a browser shows these files -- the same
files, not two renderings of one source -- so the two cannot differ. They
are built on first use into the cache (~/.cache/harwood/docs/<version>) and
again whenever harwood or a page changes.

Three kinds of page:

``written``    docsite/pages/**.md: the guides. Markdown, with three additions:
               fenced ``example`` blocks (docsite/examples.py), fenced
               ``note``/``tip``/``warning`` blocks, and links written
               ``task:daofind``, ``page:concepts/outputs`` or
               ``package:noao.digiphot.apphot``, checked here.
``tasks``      one per harwood task, made from its model and its place in the
               tree, with its manual from docsite/pages/tasks/<name>.md (or
               the task's docstring) in the middle.
``packages``   one per package of IRAF's tree, as the Tasks panel walks it.

A link to nothing, or an example harwood would refuse, stops the build with
the page and the reason: a broken page is not something to find by reading.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import posixpath
import re
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.i18n import LANGUAGES, N_, NAMES, tr, trf
from . import examples as ex

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
#: The written pages in another language, translations/<language>/<page>.md, each
#: beginning with <!-- from: <the first 12 of the English page's sha256> -->.
TRANSLATIONS = HERE / "translations"
ASSETS = HERE / "assets"
FONTS = HERE.parent / "data" / "fonts"

#: Admonitions, with a symbol and a word each: never colour alone.
CALLOUTS = {"note": ("●", N_("Note")), "tip": ("✦", N_("Tip")),
            "warning": ("▲", N_("Warning")), "problem": ("✖", N_("Problem"))}

#: The language the site is being built in (build() sets it for each one).
_LANG = "en"


def _t(text: str) -> str:
    """A text of the site's own, in the language being built."""
    return tr(text, _LANG)


def _tf(text: str, *values) -> str:
    return trf(text, *values, code=_LANG)


def source_stamp(name: str) -> str:
    """What a translation of the written page `name` records it was made from."""
    return hashlib.sha256((PAGES / name).read_bytes()).hexdigest()[:12]


def translation(name: str, code: str) -> Path:
    return TRANSLATIONS / code / name


def _source(name: str) -> tuple[str, bool]:
    """A written page's Markdown in the language being built, and whether it is
    translated (if not, the English)."""
    english = (PAGES / name).read_text(encoding="utf-8")
    if _LANG == "en":
        return english, True
    path = translation(name, _LANG)
    if not path.exists():
        return english, False
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\A<!--.*?-->\s*", "", text, flags=re.S), True


class BuildError(Exception):
    """Pages that cannot be built, each with its page and reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("the documentation does not build:\n  " + "\n  ".join(problems))
        self.problems = problems


@dataclass
class Page:
    url: str                            #: "tasks/daofind.html", from the site's root
    title: str
    section: str
    source: str = ""                    #: Markdown, for a written page
    kind: str = "written"               #: written, task, package
    key: str = ""                       #: the task name or package path
    crumbs: list[tuple[str, str]] = field(default_factory=list)
    body: str = ""
    headings: list[tuple[int, str, str]] = field(default_factory=list)
    text: str = ""                      #: plain text, for the search
    translated: bool = True             #: False: the English, said so on the page


# ------------------------------------------------------------------ where


def location() -> Path:
    from importlib.metadata import PackageNotFoundError, version

    try:
        stamp = version("harwood")
    except PackageNotFoundError:
        stamp = "dev"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "harwood" / "docs" / stamp


def ensure(out: Path | None = None) -> Path:
    """The built site, building it first if anything changed. Returns the
    directory."""
    out = Path(out or location())
    stamp = out / ".source"
    if not stamp.exists() or stamp.read_text(encoding="utf-8") != _fingerprint():
        build(out)
    return out


def url_for(topic: str = "") -> str:
    """'daofind' -> tasks/daofind.html; 'noao.digiphot' -> packages/...;
    'concepts/outputs' -> concepts/outputs.html; '' -> index.html."""
    from ..core.registry import exists

    topic = topic.strip()
    if not topic:
        return "index.html"
    if exists(topic):
        return f"tasks/{topic}.html"
    if (PAGES / f"{topic}.md").exists():
        return f"{topic}.html"
    from ..core import packages

    try:
        node = packages.find(topic)
    except KeyError:
        return ""
    return f"packages/{node.path or 'all'}.html"


def _fingerprint() -> str:
    """What the built pages depend on: harwood's code and the pages."""
    digest = hashlib.sha256()
    for root in (HERE.parent,):
        for path in sorted(root.rglob("*")):
            if path.suffix in (".py", ".md", ".yaml", ".css", ".js", ".po") and \
                    "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ build


class Site:
    def __init__(self) -> None:
        self.pages: list[Page] = []
        self.by_url: dict[str, Page] = {}
        self.examples: list[ex.Example] = []
        self.problems: list[str] = []
        self.nav: list[tuple[str, list[Page]]] = []

    def add(self, page: Page) -> Page:
        self.pages.append(page)
        self.by_url[page.url] = page
        return page


def collect() -> Site:
    """Every page, laid out but not rendered, in the language being built."""
    import yaml

    from ..core import packages
    from ..core.registry import list_tasks

    site = Site()
    nav = yaml.safe_load((PAGES / "nav.yaml").read_text(encoding="utf-8"))
    for section in nav:
        members = []
        for name in section["pages"]:
            text, translated = _source(name)
            title = next((line[2:].strip() for line in text.splitlines()
                          if line.startswith("# ")), Path(name).stem)
            members.append(site.add(Page(name.removesuffix(".md") + ".html", title,
                                         _t(section["section"]), text,
                                         translated=translated)))
        site.nav.append((_t(section["section"]), members))

    root = packages.tree()
    tasks = list_tasks()
    package_pages, task_pages = [], []

    def walk(node, trail):
        crumbs = trail + [(node.name, f"packages/{node.path or 'all'}.html")]
        page = site.add(Page(f"packages/{node.path or 'all'}.html",
                             node.name if node.path else _t("All packages"), _t("Tasks"),
                             kind="package", key=node.path, crumbs=trail))
        package_pages.append(page)
        for entry in node.tasks:
            if entry.available and entry.name in tasks and \
                    f"tasks/{entry.name}.html" not in site.by_url:
                task_pages.append(site.add(Page(
                    f"tasks/{entry.name}.html", entry.name, _t("Tasks"), kind="task",
                    key=entry.name, crumbs=crumbs)))
        for child in node.children:
            walk(child, crumbs)

    walk(root, [])
    site.nav.append((TASKS_NAV, package_pages[:1] + task_pages))
    return site


#: The name the tasks' part of the contents goes by inside the builder.
TASKS_NAV = "\0tasks"


def language_root(code: str) -> str:
    """Where a language's pages are, from the site's root: English at the root."""
    return "" if code == "en" else f"{code}/"


def build(out: Path | str | None = None) -> Path:
    """Write the whole site into `out`, in every language: English at the root,
    the others in pt/ and es/, the assets shared. Returns the directory."""
    global _LANG
    out = Path(out or location())
    sites = {}
    problems = []
    try:
        for code in LANGUAGES:
            _LANG = code
            site = collect()
            for page in site.pages:
                try:
                    render(page, site)
                except Exception as error:          # one bad page, said, not a traceback
                    site.problems.append(f"{language_root(code)}{page.url}: "
                                         f"{type(error).__name__}: {error}")
            problems += site.problems
            sites[code] = site
    finally:
        _LANG = "en"
    if problems:
        raise BuildError(problems)

    if out.exists():
        shutil.rmtree(out)
    (out / "assets" / "fonts").mkdir(parents=True)
    for asset in ASSETS.iterdir():
        shutil.copy(asset, out / "assets" / asset.name)
    for font in FONTS.iterdir():
        shutil.copy(font, out / "assets" / "fonts" / font.name)
    (out / "assets" / "look.css").write_text(_look_css(), encoding="utf-8")
    try:
        for code, site in sites.items():
            _LANG = code
            home = out / language_root(code)
            home.mkdir(parents=True, exist_ok=True)
            (home / "search-index.js").write_text(
                "window.HARWOOD_INDEX = " + json.dumps(
                    [{"t": p.title, "u": p.url, "s": p.section,
                      "h": [h[2] for h in p.headings], "x": p.text[:4000]}
                     for p in site.pages], ensure_ascii=False) + ";\n", encoding="utf-8")
            order = [page for _, members in site.nav for page in members]
            for page in site.pages:
                target = home / page.url
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(_layout(page, site, order), encoding="utf-8")
    finally:
        _LANG = "en"
    (out / ".source").write_text(_fingerprint(), encoding="utf-8")
    return out


# ------------------------------------------------------------------ pages


def render(page: Page, site: Site) -> None:
    env: dict[str, Any] = {"page": page, "site": site, "ids": set()}
    if page.kind == "written":
        page.body = _md().render(page.source, env)
        if not page.translated:
            page.body = _untranslated() + page.body
    elif page.kind == "task":
        page.body = _task_body(page, site, env)
    else:
        page.body = _package_body(page, site, env)
    page.headings = env.get("headings", [])
    page.text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page.body)))


def _untranslated() -> str:
    """Said at the top of a page shown in English for want of a translation."""
    said = _t("This page has not been translated yet; it is shown in English.")
    return (f'<div class="callout note" role="note" lang="{_LANG}"><p class="callout-title">'
            f'<span aria-hidden="true">●</span> {html.escape(_t("Not translated yet"))}</p>'
            f"<p>{html.escape(said)}</p></div>")


def _md():
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    md.add_render_rule("fence", _fence)
    md.add_render_rule("heading_open", _heading_open)
    md.add_render_rule("heading_close", _heading_close)
    md.add_render_rule("link_open", _link_open)
    md.add_render_rule("table_open", lambda self, t, i, o, e: '<div class="table"><table>')
    md.add_render_rule("table_close", lambda self, t, i, o, e: "</table></div>")
    return md


def _slug(text: str, env: dict) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
    slug, n = base, 2
    while slug in env["ids"]:
        slug, n = f"{base}-{n}", n + 1
    env["ids"].add(slug)
    return slug


def _heading_open(self, tokens, idx, options, env):
    token = tokens[idx]
    text = tokens[idx + 1].content
    slug = _slug(text, env)
    env["_heading"] = slug
    level = int(token.tag[1])
    if level in (2, 3):
        env.setdefault("headings", []).append((level, slug, _plain(text)))
    return f'<{token.tag} id="{slug}">'


def _heading_close(self, tokens, idx, options, env):
    token = tokens[idx]
    if token.tag == "h1":
        return "</h1>"
    return (f'<a class="anchor" href="#{env["_heading"]}" aria-label="Link to this '
            f'section">#</a></{token.tag}>')


def _link_open(self, tokens, idx, options, env):
    token = tokens[idx]
    href = token.attrGet("href") or ""
    resolved = _resolve(href, env)
    if resolved is not None:
        token.attrSet("href", resolved)
    elif href.startswith(("http://", "https://")):
        token.attrSet("rel", "noopener")
    return self.renderToken(tokens, idx, options, env)


def _resolve(href: str, env: dict) -> str | None:
    """task:, page: and package: links, as paths from this page; a link to
    nothing is a problem for the build."""
    from ..core import packages
    from ..core.registry import exists

    page, site = env["page"], env["site"]
    kind, _, target = href.partition(":")
    anchor = ""
    if "#" in target:
        target, anchor = target.split("#", 1)
        anchor = "#" + anchor
    if kind == "task":
        url = f"tasks/{target}.html" if exists(target) else ""
    elif kind == "page":
        url = f"{target}.html" if f"{target}.html" in site.by_url else ""
    elif kind == "package":
        try:
            url = f"packages/{packages.find(target).path or 'all'}.html"
        except KeyError:
            url = ""
    else:
        return None
    if not url:
        site.problems.append(f"{page.url}: a link to {href}, which is not there")
        return "#"
    return _relative(url, page.url) + anchor


def _relative(url: str, start: str) -> str:
    return posixpath.relpath(url, posixpath.dirname(start) or ".")


def _fence(self, tokens, idx, options, env):
    token = tokens[idx]
    info, content = token.info.strip(), token.content
    if info == "example":
        return _example(content, env)
    if info in CALLOUTS:
        symbol, word = CALLOUTS[info]
        inner = _md().render(content, {**env, "ids": env["ids"]})
        return (f'<div class="callout {info}" role="note"><p class="callout-title">'
                f'<span aria-hidden="true">{symbol}</span> {_t(word)}</p>{inner}</div>')
    if info == "generated":
        return _generated(content.strip(), env)
    return _code(content, info or "text")


def _code(code: str, language: str) -> str:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound

    try:
        lexer = get_lexer_by_name(language)
    except ClassNotFound:
        lexer = get_lexer_by_name("text")
    body = highlight(code, lexer, HtmlFormatter(nowrap=True))
    return (f'<div class="code"><button class="copy" type="button" '
            f'aria-label="{_t("Copy this")}">{_t("Copy")}</button>'
            f'<pre class="highlight lang-{language}">'
            f"<code>{body}</code></pre></div>")


def _example(content: str, env: dict) -> str:
    page, site = env["page"], env["site"]
    try:
        example = ex.parse(content, page.url)
        python = ex.python(example.lines)
        clicks = ex.clicks(example.lines)
    except ValueError as error:
        site.problems.append(f"{page.url}: an example: {error}")
        return ""
    site.examples.append(example)
    number = len(site.examples)
    terminal = _code("\n".join(f"$ {line}" for line in example.lines), "console")
    if example.output:
        terminal += (f'<p class="prints">{_t("It prints, among the rest:")}</p>'
                     f'<pre class="output"><code>{html.escape(example.output)}</code></pre>')
    if python is None:
        python_html = ("<p>" + _t("Nothing to call from Python here: this is a command for "
                                  "reading, in the terminal or at the window's prompt.")
                       + "</p>")
    else:
        python_html = _code(python, "python")
    if clicks is None:
        clicks_html = ("<p>" + _t("Type it at the window's prompt (Command line, Ctrl+L): "
                                  "it is a command for reading, not a task with a form.")
                       + "</p>")
    else:
        parts = []
        for line, steps in zip(example.lines, clicks, strict=True):
            items = "".join(f"<li>{html.escape(_t(step))}</li>" for step in steps)
            head = ('<p class="for">' + _tf("For {0}:", f"<code>{html.escape(line)}</code>")
                    + "</p>" if len(clicks) > 1 else "")
            parts.append(f"{head}<ol>{items}</ol>")
        clicks_html = "".join(parts)
    start = _relative("start/sample.html", page.url)
    tabs = "".join(
        f'<button type="button" role="tab" id="ex{number}-{way}" '
        f'aria-controls="ex{number}-{way}-panel" '
        f'aria-selected="{str(way == "terminal").lower()}" '
        f'data-way="{way}" tabindex="{0 if way == "terminal" else -1}">{label}</button>'
        for way, label in (("terminal", _t("Terminal")), ("python", "Python"),
                           ("clicks", _t("Clicks"))))
    panels = "".join(
        f'<div role="tabpanel" id="ex{number}-{way}-panel" aria-labelledby="ex{number}-{way}" '
        f'data-way="{way}"{"" if way == "terminal" else " hidden"}>{body}</div>'
        for way, body in (("terminal", terminal), ("python", python_html),
                          ("clicks", clicks_html)))
    title = f" · {html.escape(example.title)}" if example.title else ""
    route = ex.path_to(example.after)
    there = ""
    if route:
        there = (f"<details class=\"there\"><summary>{_t('How to get there')}</summary>"
                 f"<p>{_t('From the sample night, as written:')}</p>"
                 + _code("\n".join(f"$ {line}" for line in route), "console") +
                 "</details>")
    word = _t("Example")
    return (f'<section class="example" aria-label="{word}{title}">'
            f'<p class="example-title"><span aria-hidden="true">▶</span> {word}{title}</p>'
            '<p class="example-start">'
            + _tf("Starts from {0} ({1}).", _t(ex.STARTS[example.after]),
                  f'<a href="{start}">{_t("the sample night")}</a>')
            + (" " + _t("It fails, on purpose: this is what failing looks like.")
               if example.fails else "") + f"</p>{there}"
            f'<div class="tabs" role="tablist" aria-label="{_t("Ways to do it")}">{tabs}</div>'
            f"{panels}</section>")


def _generated(what: str, env: dict) -> str:
    if what == "migration-table":
        return _migration_table(env)
    if what == "task-index":
        return _task_index(env)
    if what == "stages":
        parts = []
        for said, commands in ex.STAGES.values():
            said = _t(said)
            parts.append(f"<li><b>{html.escape(said[0].upper() + said[1:])}</b>"
                         + (_code("\n".join(f"$ {c}" for c in commands), "console")
                            if commands
                            else " -- " + _tf("what {0} writes.",
                                              "<code>harwood docs --sample</code>"))
                         + "</li>")
        return f'<ol class="stages">{"".join(parts)}</ol>'

    env["site"].problems.append(f"{env['page'].url}: nothing is generated as '{what}'")
    return ""


def _plain(text: str) -> str:
    return re.sub(r"[`*_]", "", text)


# ------------------------------------------------------------ task pages


def _task_body(page: Page, site: Site, env: dict) -> str:
    from ..cli.main import _show, _task_command, _typename
    from ..core import migration
    from ..core.registry import get_class

    cls = get_class(page.key)
    md = _md()
    parts = [f"<h1>{html.escape(cls.name)}</h1>",
             f'<p class="lead">{html.escape(_t(cls.description))}</p>']
    facts = [_tf("package {0}", html.escape(".".join(c[0] for c in page.crumbs[1:])
                                             or "all"))]
    iraf = cls.iraf or cls.name
    if migration.lookup(iraf):
        facts.append(f"IRAF: {html.escape(iraf)}"
                     + (" " + _t("(the same name)") if iraf == cls.name else ""))
    if cls.defaults:
        facts.append(_t("a preset:") + " " + html.escape(", ".join(
            f"{k}={v}" for k, v in cls.defaults.items())))
    if cls.needs_window:
        facts.append(_t("works in the window"))
    parts.append('<p class="facts">' + "".join(f"<span>{f}</span>" for f in facts) + "</p>")

    manual = PAGES / "tasks" / f"{cls.name}.md"
    if manual.exists():
        text, translated = _source(f"tasks/{cls.name}.md")
        if not translated:
            parts.append(_untranslated())
        parts.append(md.render(text, env))
    elif cls.__doc__ and cls.__doc__.strip():
        text = textwrap.dedent("\n".join(cls.__doc__.splitlines()[1:]))
        first = cls.__doc__.splitlines()[0].strip()
        parts.append(md.render(f"{first}\n\n{text}".strip(), env))
    else:
        parts.append(md.render(
            "```note\n" + _tf("This task has no long manual yet. Its parameters below are "
                              "complete -- each one says what it does -- and `harwood "
                              "{0} --plan` shows what a run would do, step by step, "
                              "before it does it.", cls.name) + "\n```", env))

    command = _task_command(cls.name)
    usage = " ".join(command.collect_usage_pieces(_context(command)))
    link = _t("Link to this section")
    env.setdefault("headings", []).append((2, "command-line", _t("Command line")))
    env["ids"].add("command-line")
    parts.append(f'<h2 id="command-line">{_t("Command line")}<a class="anchor" '
                 f'href="#command-line" aria-label="{link}">#</a></h2>')
    parts.append(_code(f"harwood {cls.name} {usage}", "text"))
    parts.append(md.render(_tf(
        "Every parameter below is also an option, `--name value`; "
        "`harwood help {0}` prints this page's table in the terminal, and "
        "`harwood lpar {0}` the values a run would actually use, and where "
        "each came from.", cls.name), env))

    env["headings"].append((2, "parameters", _t("Parameters")))
    env["ids"].add("parameters")
    rows = []
    for name, info in cls.Params.model_fields.items():
        required = info.is_required()
        positional = required or (info.json_schema_extra or {}).get("positional")
        default = _t("required") if required else _show(info.get_default(
            call_default_factory=True))
        how = _t("positional") if positional else f"--{name.replace('_', '-')}"
        rows.append(
            f'<tr id="param-{name}"><th scope="row"><code>{html.escape(name)}</code></th>'
            f"<td>{html.escape(how)}</td><td><code>{html.escape(_typename(info.annotation))}"
            f"</code></td><td>{html.escape(str(default))}</td>"
            f"<td>{html.escape(info.description or '')}</td></tr>")
    heads = "".join(f"<th>{_t(h)}</th>" for h in (
        N_("parameter"), N_("written"), N_("type"), N_("default"), N_("what it does")))
    parts.append(f'<h2 id="parameters">{_t("Parameters")}<a class="anchor" '
                 f'href="#parameters" aria-label="{link}">#</a></h2>'
                 + ("" if _LANG == "en" else '<p class="hint">' + _t(
                     "The descriptions of the parameters are in English, as harwood "
                     "keeps them everywhere.") + "</p>")
                 + '<div class="table"><table class="params"><thead><tr>'
                 f"{heads}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")

    entry = migration.lookup(iraf)
    if entry:
        env["headings"].append((2, "in-iraf", _t("In IRAF")))
        env["ids"].add("in-iraf")
        note = " ".join(str(entry.get("note", "")).split())
        status = entry.get("status", "")
        said = {"available": N_("harwood has it, complete."),
                "partial": N_("harwood has it, with gaps named below."),
                }.get(status, status)
        parts.append(f'<h2 id="in-iraf">{_t("In IRAF")}<a class="anchor" href="#in-iraf" '
                     f'aria-label="{link}">#</a></h2>'
                     "<p>" + _tf("IRAF calls it {0}.", f"<code>{html.escape(iraf)}</code>")
                     + f" {html.escape(_t(said))}"
                     + (f" {html.escape(note)}" if note else "") + "</p>")

    siblings = [p for p in site.pages if p.kind == "task" and p.crumbs == page.crumbs
                and p is not page]
    if siblings:
        env["headings"].append((2, "same-package", _t("In the same package")))
        env["ids"].add("same-package")
        links = "".join(f'<li><a href="{_relative(p.url, page.url)}">{p.title}</a></li>'
                        for p in siblings)
        parts.append(f'<h2 id="same-package">{_t("In the same package")}<a class="anchor" '
                     f'href="#same-package" aria-label="{link}">#</a></h2>'
                     f'<ul class="inline">{links}</ul>')
    return "".join(parts)


def _context(command):
    import click

    return click.Context(command, info_name=command.name)


# ---------------------------------------------------------- package pages


def _package_body(page: Page, site: Site, env: dict) -> str:
    from ..core import packages

    node = packages.find(page.key)
    title = node.name if node.path else _t("All packages")
    parts = [f"<h1>{html.escape(title)}</h1>",
             f'<p class="lead">{html.escape(_t(node.brief))}</p>']
    if node.children:
        env.setdefault("headings", []).append((2, "packages", _t("Packages")))
        cards = []
        for child in node.children:
            have, total = child.available_count(), child.total_count()
            muted = ' class="muted"' if not have else ""
            cards.append(
                f'<li{muted}><a href="{_relative(f"packages/{child.path}.html", page.url)}">'
                f'<span class="icon" aria-hidden="true">{html.escape(child.icon)}</span>'
                f"<b>{html.escape(child.name)}</b></a> <span class=\"count\">"
                + _tf("{0} of {1} task in harwood" if total == 1
                      else "{0} of {1} tasks in harwood", have, total)
                + f"</span><span class=\"brief\">{html.escape(_t(child.brief))}</span></li>")
        parts.append(f'<h2 id="packages">{_t("Packages")}</h2>'
                     f'<ul class="cards">{"".join(cards)}</ul>')
    here = [t for t in node.tasks if t.available]
    missing = [t for t in node.tasks if not t.available]
    if here:
        env.setdefault("headings", []).append((2, "tasks", _t("Tasks")))
        items = "".join(
            f'<li><a href="{_relative(f"tasks/{t.name}.html", page.url)}"><b>{t.name}</b>'
            f"</a> <span class=\"brief\">{html.escape(_t(t.brief))}</span></li>"
            for t in here)
        parts.append(f'<h2 id="tasks">{_t("Tasks")}</h2><ul class="cards">{items}</ul>')
    if missing:
        words = {"planned": _t("planned"), "replaced": _t("replaced"),
                 "wontport": _t("not ported"), "partial": _t("partial")}
        rows = "".join(
            f"<li><b>{html.escape(t.iraf)}</b> <span class=\"status\">"
            f"{words.get(t.status, t.status)}</span>"
            + (f' <span class="brief">{html.escape(t.note)}</span>' if t.note else "")
            + "</li>" for t in missing)
        parts.append("<details><summary>"
                     + _tf("IRAF tasks here that harwood does not have ({0})", len(missing))
                     + f"</summary><ul class=\"missing\">{rows}</ul></details>")
    return "".join(parts)


def _migration_table(env: dict) -> str:
    from ..core import migration
    from ..core.registry import exists

    words = {"available": _t("in harwood"), "partial": _t("in harwood, with gaps"),
             "planned": _t("planned"), "replaced": _t("replaced"),
             "wontport": _t("not ported")}
    rows = []
    for name, entry in sorted({**migration.tasks(), **migration.added()}.items()):
        target = entry.get("harwood", name)
        status = entry.get("status", "planned")
        package = entry["pkg"] if isinstance(entry["pkg"], str) else ", ".join(entry["pkg"])
        cell = (f'<a href="{_relative(f"tasks/{target}.html", env["page"].url)}">{target}</a>'
                if exists(target) else html.escape(target if target != name else "—"))
        note = " ".join(str(entry.get("note", "")).split())
        rows.append(f'<tr data-status="{status}"><th scope="row">{html.escape(name)}</th>'
                    f"<td>{html.escape(package)}</td><td>{cell}</td>"
                    f"<td>{words.get(status, status)}</td><td>{html.escape(note)}</td></tr>")
    return (f'<p><label>{_t("Find a task:")} <input type="search" class="filter" '
            'data-filter="migration" placeholder="imcombine, phot, splot..."></label></p>'
            '<div class="table"><table class="migration" id="migration"><thead><tr>'
            f"<th>IRAF</th><th>{_t('package')}</th><th>harwood</th><th>{_t('status')}</th>"
            f"<th>{_t('note')}</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _task_index(env: dict) -> str:
    from ..core.registry import list_tasks

    items = "".join(
        f'<li><a href="{_relative(f"tasks/{name}.html", env["page"].url)}"><b>{name}</b>'
        f'</a> <span class="brief">{html.escape(_t(cls.description))}</span></li>'
        for name, cls in sorted(list_tasks().items()))
    return f'<ul class="cards">{items}</ul>'


# ------------------------------------------------------------------ layout


def _languages(page: Page, root: str) -> str:
    """The same page in each language: EN PT ES, the one shown marked."""
    top = root + ("../" if _LANG != "en" else "")
    links = []
    for code in LANGUAGES:
        href = f"{top}{language_root(code)}{page.url}"
        current = ' aria-current="true"' if code == _LANG else ""
        links.append(f'<a href="{href}" hreflang="{code}" lang="{code}" '
                     f'title="{NAMES[code]}"{current}>{code.upper()}</a>')
    return (f'<nav class="langs" aria-label="{_t("Language")} · Idioma">'
            f'<span aria-hidden="true">🌐</span> {"".join(links)}</nav>')


#: What the page's script says, in the page's language.
_SCRIPT_WORDS = {"copy": N_("Copy"), "copied": N_("Copied ✓"),
                 "nothing": N_("Nothing found for “{0}”.")}


def _layout(page: Page, site: Site, order: list[Page]) -> str:
    root = _relative("index.html", page.url).removesuffix("index.html") or "./"
    # the assets are the site's, shared by every language: above a translation
    base = root + ("../" if _LANG != "en" else "")
    words = json.dumps({k: _t(v) for k, v in _SCRIPT_WORDS.items()}, ensure_ascii=False)
    nav = []
    for section, members in site.nav:
        if section == TASKS_NAV:
            nav.append(_tasks_nav(page, site))
            continue
        items = "".join(
            f'<li><a href="{_relative(p.url, page.url)}"'
            f'{" aria-current=page" if p is page else ""}>{html.escape(p.title)}</a></li>'
            for p in members)
        nav.append(f'<p class="nav-section">{html.escape(section)}</p><ul>{items}</ul>')
    toc = "".join(
        f'<li class="level{level}"><a href="#{slug}">{html.escape(text)}</a></li>'
        for level, slug, text in page.headings)
    crumbs = ""
    if page.crumbs:
        crumbs = f'<nav class="crumbs" aria-label="{_t("Where this is")}">' + " › ".join(
            f'<a href="{_relative(url, page.url)}">{html.escape(name)}</a>'
            for name, url in page.crumbs) + "</nav>"
    back = ""
    if page.crumbs:
        name, url = page.crumbs[-1]
        back = (f'<p class="back"><a href="{_relative(url, page.url)}">'
                + _tf("◀ Back to {0}", html.escape(name)) + "</a></p>")
    contents = f"<p>{_t('On this page')}</p><ul>{toc}</ul>" if toc else ""
    position = order.index(page) if page in order else -1
    pager = ""
    if position >= 0:
        before = order[position - 1] if position > 0 else None
        after = order[position + 1] if position + 1 < len(order) else None
        pager = f'<nav class="pager" aria-label="{_t("Next and previous")}">' + (
            f'<a class="prev" href="{_relative(before.url, page.url)}">← '
            f"{html.escape(before.title)}</a>" if before else "<span></span>") + (
            f'<a class="next" href="{_relative(after.url, page.url)}">'
            f"{html.escape(after.title)} →</a>" if after else "") + "</nav>"
    return f"""<!doctype html>
<html lang="{_LANG}" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(page.title)} · harwood</title>
<link rel="stylesheet" href="{base}assets/look.css">
<link rel="stylesheet" href="{base}assets/site.css">
<script src="{base}assets/look.js"></script>
<script>window.HARWOOD_WORDS = {words};</script>
</head>
<body data-root="{root}">
<a class="skip" href="#main">{_t("Skip to the page")}</a>
<header class="top">
  <button class="menu" type="button" aria-label="{_t("Contents")}"
          aria-expanded="false">☰</button>
  <a class="brand" href="{root}index.html"><span aria-hidden="true">✦</span> harwood</a>
  <div class="search" role="search">
    <input type="search" id="search" placeholder="{_t("Search the documentation")}  /"
           aria-label="{_t("Search the documentation")}" autocomplete="off">
    <ul class="results" id="results" role="listbox" hidden></ul>
  </div>
  {_languages(page, root)}
  <div class="look" role="group" aria-label="{_t("How the page looks")}">
    <button type="button" data-look="theme"
            aria-label="{_t("Light or dark")}">◐ <span>{_t("theme")}</span></button>
    <button type="button" data-look="colorblind"
            aria-pressed="false">◍ <span>{_t("colour-blind")}</span></button>
    <button type="button" data-look="dyslexia"
            aria-pressed="false">Aa <span>{_t("dyslexia")}</span></button>
  </div>
</header>
<div class="layout">
<nav class="side" aria-label="{_t("Contents")}">{"".join(nav)}</nav>
<main id="main" tabindex="-1">
{crumbs}{back}
<article>{page.body}</article>
{pager}
<footer class="foot">{_FOOT()}</footer>
</main>
<aside class="toc" aria-label="{_t("On this page")}">{contents}</aside>
</div>
<script src="{root}search-index.js"></script>
<script src="{base}assets/site.js"></script>
</body>
</html>
"""


def _FOOT() -> str:  # noqa: N802 -- read as a piece of the page
    return _tf("harwood documentation · the same pages in the window (Help › "
               "Documentation, F1) and in a browser ({0}).",
               "<code>harwood docs --browser</code>")


def _tasks_nav(page: Page, site: Site) -> str:
    """The package tree, opened along the way to this page."""
    from ..core import packages

    root = packages.tree()
    here = page.key if page.kind == "package" else ""
    if page.kind == "task" and page.crumbs:
        here = page.crumbs[-1][1].removeprefix("packages/").removesuffix(".html")

    def branch(node) -> str:
        if not node.available_count():
            return ""
        tasks = "".join(
            f'<li><a href="{_relative(f"tasks/{t.name}.html", page.url)}"'
            f'{" aria-current=page" if page.kind == "task" and page.key == t.name else ""}>'
            f"{t.name}</a></li>" for t in node.tasks if t.available)
        children = "".join(branch(c) for c in node.children)
        open_ = " open" if here == node.path or here.startswith(node.path + ".") else ""
        url = _relative(f"packages/{node.path}.html", page.url)
        current = " aria-current=page" if page.kind == "package" and page.key == node.path \
            else ""
        return (f"<li><details{open_}><summary><a href=\"{url}\"{current}>{node.name}</a>"
                f"</summary><ul>{children}{tasks}</ul></details></li>")

    tree = "".join(branch(child) for child in root.children)
    everything = _relative("packages/all.html", page.url)
    return (f'<p class="nav-section">{_t("Tasks")}</p><ul><li><a href="{everything}">'
            f'{_t("All packages")}</a></li>{tree}</ul>')


def _look_css() -> str:
    """The window's colours, as the page's: the same tokens, so that the two
    look alike in every mode."""
    from dataclasses import replace

    from ..viewer import theme

    def block(selector: str, palette) -> str:
        names = ("ground", "panel", "line", "ink", "muted", "accent", "sky", "comment",
                 "note", "warning", "problem")
        body = " ".join(f"--{n}: {getattr(palette, n)};" for n in names)
        return f"{selector} {{ {body} }}\n"

    css = block(":root, html[data-theme=dark]", theme.DARK)
    css += block("html[data-theme=light]", theme.LIGHT)
    css += block("html[data-theme=dark][data-colorblind]", theme.DARK_CB)
    css += block("html[data-theme=light][data-colorblind]", theme.LIGHT_CB)
    css += block("html[data-theme=light][data-dyslexia]",
                 replace(theme.LIGHT, ground=theme.CREAM))
    css += block("html[data-theme=light][data-dyslexia][data-colorblind]",
                 replace(theme.LIGHT_CB, ground=theme.CREAM))
    return css
