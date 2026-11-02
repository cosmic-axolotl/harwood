"""What harwood declares it needs, against what it actually imports.

This exists because `harwood` was, for some weeks, not installable at all: the
command line had been moved from typer to click and the dependency list still
said typer. typer used to pull click in, so it worked in the development
environment and nowhere else. Nobody would have found out until someone ran
`pip install harwood`, which is a bad way to find out.

The check also enforces the rule in section 3.2 that keeps `list` and `--help`
fast. An import at module scope has to be a required dependency; anything
optional has to be imported inside the function that needs it, which is also
what stops importing a task from pulling in astropy.
"""

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "harwood"
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

#: Import name -> the distribution that provides it, where they differ.
DISTRIBUTION = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "PySide6": "pyside6-essentials",
    # PySide6's runtime, which pyside6-essentials brings with it
    "shiboken6": "pyside6-essentials",
    "numba": "numba",
    "harwood_kernels": "harwood-kernels",
    "markdown_it": "markdown-it-py",
    "pygments": "pygments",
}

#: Modules that nothing imports unless the feature is present -- the viewer is
#: reached only through the display interface, and numba_impl only through the
#: kernel dispatcher, inside a try/ImportError. These may import their extra at
#: module scope; everything else may not.
CONDITIONAL = ("viewer/", "kernels/numba_impl.py", "kernels/rust_impl.py")

#: Only these may see Qt at all.
GUI_ONLY = ("viewer",)


def conditional(relative) -> bool:
    text = relative.as_posix()
    return any(text.startswith(prefix) or text == prefix for prefix in CONDITIONAL)


def declared() -> tuple[set[str], set[str]]:
    """(required, optional) distribution names from pyproject."""
    import tomllib

    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = document["project"]

    def names(entries):
        found = set()
        for entry in entries:
            name = entry.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
            found.add(name.strip().lower())
        return found

    required = names(project.get("dependencies", []))
    optional = set()
    for group in project.get("optional-dependencies", {}).values():
        optional |= names(group)
    return required, optional


def module_scope_imports(path: Path) -> set[str]:
    """Third-party modules imported at module scope, not inside a function."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in tree.body:                      # top level only, on purpose
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` costs nothing at runtime.
            for inner in node.body:
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    continue
    return {name for name in found
            if name not in sys.stdlib_module_names and name != "harwood"}


def every_import(path: Path) -> set[str]:
    """Third-party modules imported anywhere in a file, at any depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return {name for name in found
            if name not in sys.stdlib_module_names and name != "harwood"}


def sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_every_module_scope_import_is_declared():
    required, optional = declared()
    problems = []
    for path in sources():
        relative = path.relative_to(PACKAGE)
        lenient = conditional(relative)
        allowed = required | optional if lenient else required
        for name in module_scope_imports(path):
            distribution = DISTRIBUTION.get(name, name).lower()
            if distribution not in allowed:
                problems.append(
                    f"{relative} imports {name} at module scope "
                    f"({distribution} is not declared"
                    + ("" if lenient else " as a required dependency")
                    + "). Declare it, or import it inside the function that "
                      "needs it.")
    assert not problems, "\n".join(problems)


def test_nothing_outside_the_viewer_imports_qt_at_module_scope():
    """Section 4 wants a harwood-core that installs without a GUI on a server.

    Qt is the largest dependency by far, and one import of it at the wrong
    level makes the whole package need it.
    """
    offenders = [
        path.relative_to(PACKAGE) for path in sources()
        if path.relative_to(PACKAGE).parts[0] not in GUI_ONLY
        and {"PySide6", "pyqtgraph"} & module_scope_imports(path)
    ]
    assert not offenders, f"Qt imported at module scope in {offenders}"


def test_the_declared_dependencies_are_all_used():
    """A dependency nobody imports is one that will be wrong before it is
    noticed, which is exactly how typer outlived its use here."""
    required, _ = declared()
    used = set()
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                used |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                used.add(node.module.split(".")[0])
    used = {DISTRIBUTION.get(name, name).lower() for name in used}
    unused = required - used - {"numpy"}        # numpy comes in through astropy's API
    assert not unused, f"declared but never imported: {sorted(unused)}"


@pytest.mark.parametrize("command", ["list", "doctor"])
def test_the_commands_that_must_be_instant_do_not_import_astropy(command):
    """Section 3.2: `list` and `--help` answer instantly, which they cannot do
    if reaching a task pulls in astropy."""
    import subprocess

    code = (
        "import sys, runpy, io, contextlib;"
        f"sys.argv = ['harwood', {command!r}];"
        "out = io.StringIO();"
        "exec('try:\\n"
        " with contextlib.redirect_stdout(out):\\n"
        "  runpy.run_module(\"harwood.cli.main\", run_name=\"__main__\")\\n"
        "except SystemExit:\\n pass');"
        "print('astropy' in sys.modules, 'PySide6' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-800:]
    astropy_loaded, qt_loaded = result.stdout.strip().split()
    assert astropy_loaded == "False", f"`harwood {command}` imported astropy"
    assert qt_loaded == "False", f"`harwood {command}` imported Qt"


def test_nothing_is_imported_that_nobody_declared_anywhere():
    """The module-scope check above misses an import inside a function, and
    that is where the awkward ones hide: the headless display reached for an
    imaging library that is in no extra at all, on the one code path that runs
    where the GUI extra was deliberately not installed."""
    required, optional = declared()
    known = required | optional
    problems = []
    for path in sources():
        for name in every_import(path):
            distribution = DISTRIBUTION.get(name, name).lower()
            if distribution not in known:
                problems.append(f"{path.relative_to(PACKAGE)} imports {name}, "
                                f"which no dependency or extra provides")
    assert not problems, "\n".join(problems)


def test_the_headless_display_writes_a_real_png(tmp_path):
    """On a server with no GUI extra, the one code path that has to produce a
    picture cannot be the one that needs an imaging library."""
    import numpy as np

    from harwood.viewer.png import write_grey

    gradient = np.outer(np.linspace(0, 255, 24), np.ones(32))
    path = write_grey(tmp_path / "x.png", gradient)
    raw = path.read_bytes()

    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in raw and b"IDAT" in raw and raw.endswith(b"IEND\xae\x42\x60\x82")

    # Read it back with something that is not the writer.
    import struct
    import zlib

    width, height, depth, colour = struct.unpack(">IIBB", raw[16:26])
    assert (width, height, depth, colour) == (32, 24, 8, 0)
    start = raw.index(b"IDAT") + 4
    length = struct.unpack(">I", raw[start - 8:start - 4])[0]
    rows = zlib.decompress(raw[start:start + length])
    assert len(rows) == height * (width + 1), "one filter byte per row"
    assert rows[0] == 0, "no filtering"


def test_a_two_dimensional_array_is_required():
    import numpy as np
    import pytest

    from harwood.viewer.png import write_grey

    with pytest.raises(ValueError, match="2-D"):
        write_grey("/tmp/never-written.png", np.zeros((2, 2, 3)))


def test_a_missing_extra_is_named_with_the_command_that_installs_it():
    from harwood.core import extras

    assert "harwood[phot]" in extras.hint(ImportError(name="scipy"))
    assert "harwood[gui]" in extras.hint(ImportError(name="PySide6"))
    assert "harwood[fast]" in extras.hint(ImportError(name="numba"))
    assert extras.hint(ImportError(name="a_real_bug")) is None, \
        "inventing an install command for a real bug sends people the wrong way"
    assert extras.hint(ValueError("not an import")) is None


def test_every_optional_module_the_code_reaches_for_is_in_the_table():
    """extras.PROVIDED_BY is what turns a missing import into an instruction;
    a module missing from it gets the unhelpful message instead."""
    required, _ = declared()
    from harwood.core import extras

    reached = set()
    for path in sources():
        reached |= every_import(path)
    optional_modules = {
        name for name in reached
        if DISTRIBUTION.get(name, name).lower() not in required
        and name != "harwood_kernels"
    }
    missing = optional_modules - set(extras.PROVIDED_BY)
    assert not missing, f"not in extras.PROVIDED_BY: {sorted(missing)}"


# ------------------------------------------------- no IRAF anywhere


@pytest.fixture
def no_iraf(monkeypatch):
    """A machine where IRAF is not installed and cannot be found -- as if it
    had disappeared. Only harwood's own copy of its tables is left."""
    from harwood.core import arc, flux, instrument

    monkeypatch.setenv("iraf", "/nonexistent/iraf")
    for module, name in ((arc, "IRAF_LINELISTS"), (flux, "IRAF_STANDARDS"),
                         (instrument, "IRAF_CCDDB")):
        own = getattr(module, name)[0]
        assert "site-packages" in str(own) or "src/harwood/data/iraf" in str(own)
        monkeypatch.setattr(module, name, (own,))


def test_harwood_never_imports_iraf_or_pyraf():
    """harwood is a reimplementation: IRAF is the reference it is checked
    against, never a library it calls."""
    import re

    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+(pyraf|iraf)\b", text, re.M), path


def test_iraf_tables_work_without_iraf(no_iraf, tmp_path):
    """linelists$, onedstds$ and ccddb$ are IRAF's names for its tables, and
    harwood ships the tables: written the IRAF way, they resolve with no IRAF
    on the machine."""
    from harwood.core import arc, flux, identify, instrument

    lines = identify.read_linelist(arc.resolve_linelist("linelists$thar.dat"))
    assert len(lines) > 3000
    assert flux.resolve("onedstds$kpnoextinct.dat").is_file()
    assert flux.find_star("bd284211", "spec50cal") is not None
    assert "spec50cal" in [c for c in flux.CALDIRS if flux.stars(c)]
    direct = instrument.load("ccddb$kpno/direct.dat")
    assert direct.keyword("subset") == "FILTERS" or direct.types
    assert "direct" in instrument.installed()["kpno"]


def test_the_iraf_tables_are_in_the_package(tmp_path):
    """Shipped means in the wheel, not only in the source tree."""
    import subprocess
    import sys
    import zipfile

    built = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q",
                            "-w", str(tmp_path), str(PYPROJECT.parent)],
                           capture_output=True, text=True)
    if built.returncode != 0:
        pytest.skip(f"cannot build a wheel here: {built.stderr[-200:]}")
    [wheel] = tmp_path.glob("harwood-*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    for table in ("harwood/data/iraf/linelists/thar.dat",
                  "harwood/data/iraf/onedstds/kpnoextinct.dat",
                  "harwood/data/iraf/onedstds/spec50cal/bd284211.dat",
                  "harwood/data/iraf/ccddb/kpno/direct.dat",
                  "harwood/data/iraf/README",
                  "harwood/docsite/pages/index.md",
                  "harwood/docsite/assets/site.js"):
        assert table in names, f"{table} is not in the wheel"
