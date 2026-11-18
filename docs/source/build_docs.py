"""Rebuild the two harwood documents kept in ~/Documents, from the sources here.

    python docs/source/build_docs.py registro    # the project record (pandoc + xelatex)
    python docs/source/build_docs.py decisoes    # the contract: page 9 replaced (WeasyPrint)
    python docs/source/build_docs.py all

The record is written whole from registro/registro.md. The contract has no source for
sections 1 to 8 (the original PDF was written elsewhere): its page 9 -- 8.5, 9 and 10 -- is
replaced by secao-09-10-estado.html, and the addenda (11, 12) stay as they are. Whatever is
replaced is first copied to docs/historico/ with the date.

Needs: pandoc and xelatex; WeasyPrint 70 and pypdf (WEASYPRINT=/path/to/weasyprint and
PYTHON with pypdf, if not on PATH).
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DOCS = Path.home() / "Documents"
HISTORY = REPO / "docs" / "historico"
RECORD = DOCS / "harwood - registro do projeto.pdf"
CONTRACT = DOCS / "harwood-decisoes-de-projeto.pdf"
REPLACED_PAGE = 9          # 1-based: 8.5, 9 and 10 in the original


def _keep(path: Path) -> None:
    if path.exists():
        HISTORY.mkdir(parents=True, exist_ok=True)
        stamp = datetime.date.today().isoformat()
        target = HISTORY / f"{path.stem} (antes de {stamp}){path.suffix}"
        if not target.exists():
            shutil.copy2(path, target)


def registro() -> Path:
    _keep(RECORD)
    for old in DOCS.glob("harwood - registro do projeto (*).pdf"):
        _keep(old)
        old.unlink()
    subprocess.run(["pandoc", str(HERE / "registro" / "registro.md"), "-o", str(RECORD),
                    "--pdf-engine=xelatex", "--toc"], check=True)
    return RECORD


def decisoes() -> Path:
    from pypdf import PdfReader, PdfWriter

    weasy = os.environ.get("WEASYPRINT", "weasyprint")
    _keep(CONTRACT)
    original = HISTORY / sorted(p.name for p in HISTORY.glob(
        "harwood-decisoes-de-projeto (antes de *).pdf"))[0]
    with tempfile.TemporaryDirectory() as where:
        page = Path(where) / "9.pdf"
        subprocess.run([weasy, str(HERE / "secao-09-10-estado.html"), str(page)], check=True)
        old, new = PdfReader(original), PdfReader(page)
        out = PdfWriter()
        for i, p in enumerate(old.pages, start=1):
            if i == REPLACED_PAGE:
                for q in new.pages:
                    out.add_page(q)
            else:
                out.add_page(p)
        out.add_metadata({"/Title": "harwood — decisões de projeto"})
        with open(CONTRACT, "wb") as f:
            out.write(f)
    shutil.copy2(CONTRACT, REPO / "docs" / "design-decisions.pdf")
    return CONTRACT


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    done = []
    if what in ("registro", "all"):
        done.append(registro())
    if what in ("decisoes", "all"):
        done.append(decisoes())
    for path in done:
        print(path)
