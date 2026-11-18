"""The window's session: everything open, kept when harwood closes (and every two
minutes, against a crash or a power cut), and put back the next time -- or not:
`harwood gui` asks, when there is one, whether to restore it or start a new one.

What is kept: the folder it was working in; the window's size and its panels'
arrangement; each frame's pixels, header, name, z1 z2, marks and DQ, which frame was
shown, the scale, stretch and colours, the zoom and where the view was; imexamine's
parameters and what it measured; every tab of graphs (a graph as it was; implot
alive again, at its line or column; a fit whose task is over, as the picture it
was); every text file open in the editor, with what was typed and not saved; the
command line's record and what was typed at the prompt.

It lives in ~/.local/state/harwood/session (XDG_STATE_HOME): session.json, which a
person can read (when, where, what), and state.pickle and frame_<n>.npy, which
harwood reads. Starting a new session sets the old one aside as previous/, which
File > Restore previous session can still bring back until harwood closes again.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

VERSION = 1
AUTOSAVE_MS = 120_000


def folder() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "harwood" / "session"


def previous_folder() -> Path:
    return folder().parent / "session.previous"


# ------------------------------------------------------------- what is there


def summary(where: Path | None = None) -> dict | None:
    """session.json of a kept session, or None when there is none."""
    path = (where or folder()) / "session.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("version") != VERSION or not (where or folder()).joinpath(
            "state.pickle").is_file():
        return None
    return meta


def worth_restoring(meta: dict | None) -> bool:
    """Something to bring back: an image, a tab, a file, or a folder of its own."""
    if not meta:
        return False
    return bool(meta.get("frames") or meta.get("tabs") or meta.get("files")
                or meta.get("cwd") and meta["cwd"] != str(Path.home()))


def describe(meta: dict) -> list[str]:
    """The session in a few lines, for the question at the start."""
    from .lang import trf

    lines = [trf("Closed {0}, in {1}", meta.get("when", "?"), meta.get("cwd", "?"))]
    frames = meta.get("frames", [])
    if frames:
        lines.append(trf("Images: {0}", ", ".join(f"{n}: {name}" for n, name in frames)))
    if meta.get("tabs"):
        lines.append(trf("Graphs: {0}", ", ".join(meta["tabs"])))
    if meta.get("files"):
        lines.append(trf("Text files: {0}", ", ".join(Path(p).name for p in meta["files"])))
    if meta.get("unsaved"):
        lines.append(trf("Not saved yet: {0}", ", ".join(Path(p).name
                                                         for p in meta["unsaved"])))
    return lines


def set_aside() -> None:
    """A new session: the kept one becomes the previous one."""
    here, aside = folder(), previous_folder()
    if not here.exists():
        return
    shutil.rmtree(aside, ignore_errors=True)
    here.rename(aside)


# ------------------------------------------------------------- saving


def _header_text(header) -> str | None:
    if header is None:
        return None
    try:
        return header.tostring(sep="\n")
    except AttributeError:
        return None


def _header_back(text: str | None):
    if text is None:
        return None
    from astropy.io import fits

    return fits.Header.fromstring(text, sep="\n")


def _panel_state(panel) -> tuple[str, Any] | None:
    """A tab, as something that can be made again: ("graph", Graph), ("implot",
    the engine's state) or, for anything else, ("picture", a Graph of its plot)."""
    from .graph_panel import GraphPanel
    from .implot_panel import ImplotPanel

    if hasattr(panel, "save_picture"):
        return "figure", str(Path(panel.path).resolve())
    if isinstance(panel, ImplotPanel):
        state = {k: v for k, v in panel.engine.__dict__.items() if k != "read"}
        return "implot", {"name": panel.name, "engine": state}
    if isinstance(panel, GraphPanel):
        return "graph", panel.graph
    plot = getattr(panel, "plot", None)
    if plot is None:
        return None
    return "picture", picture(plot)


def picture(plot) -> Any:
    """What a pyqtgraph plot shows, as a Graph: its title, axes' labels, curves and
    points -- a fit's tab after its task has ended, to look back at."""
    import pyqtgraph as pg

    from ..core.graph import Graph

    item = plot.getPlotItem()
    title = item.titleLabel.text or ""
    labels = {}
    for side in ("bottom", "left"):
        axis = item.getAxis(side)
        labels[side] = axis.labelText if axis.label.isVisible() else ""
    graph = Graph(_plain(title), _plain(labels["bottom"]), _plain(labels["left"]))
    for data in item.listDataItems():
        x, y = data.getData()
        if x is None or y is None or not len(x):
            continue
        has_line = data.opts.get("pen") is not None and not (
            isinstance(data.opts.get("pen"), str) and data.opts["pen"] == "")
        symbol = data.opts.get("symbol")
        if has_line and symbol is None:
            graph.line(x, y)
        else:
            for a, b in zip(x, y, strict=True):
                graph.mark(float(a), float(b), "point")
            if has_line and not isinstance(data, pg.ScatterPlotItem):
                graph.line(x, y)
    (x1, x2), (y1, y2) = item.getViewBox().viewRange()
    graph.window = (x1, x2, y1, y2)
    return graph


def _plain(html: str) -> str:
    import re

    return re.sub(r"<[^>]+>", " ", html or "").replace("&nbsp;", " ").strip()


def save(window, where: Path | None = None) -> Path:
    """Keep what the window has open. Frames already written are not written
    again (their files are named by the frame and kept while the frame is the
    same object)."""
    here = where or folder()
    here.mkdir(parents=True, exist_ok=True)
    written: dict[int, tuple[int, str]] = getattr(window, "_session_frames", {})
    frames, keep = [], set()
    for number, frame in sorted(window.frames.items()):
        name = f"frame_{number}.npy"
        if written.get(number, (None,))[0] != id(frame) or not (here / name).is_file():
            np.save(here / name, frame.data, allow_pickle=False)
            if frame.quality is not None:
                np.save(here / f"dq_{number}.npy", frame.quality, allow_pickle=False)
            written[number] = (id(frame), name)
        keep.update({name, f"dq_{number}.npy"})
        frames.append({
            "number": number, "file": name, "name": frame.name,
            "folder": str(frame.folder), "header": _header_text(frame.header),
            "window": frame.window, "table": frame.table, "markers": frame.markers,
            "quality": frame.quality is not None})
    window._session_frames = written
    for stale in here.glob("*.npy"):
        if stale.name not in keep:
            stale.unlink(missing_ok=True)

    tabs, titles = [], []
    for index in range(1, window.views.count()):
        panel = window.views.widget(index)
        if panel in window._waiting:
            continue                      # a task still asking: not a thing to keep
        title = window.views.tabText(index)
        try:
            state = _panel_state(panel)
            if state is not None:
                pickle.dumps(state)
        except Exception:  # noqa: BLE001 -- a tab that cannot be kept is left out
            state = None
        if state is not None:
            tabs.append((title, *state))
            titles.append(title)

    files, unsaved = [], []
    for tab in window.files.files():
        text = tab.toPlainText() if tab.modified else None
        files.append({"path": str(tab.path), "text": text,
                      "position": tab.textCursor().position()})
        if text is not None:
            unsaved.append(str(tab.path))

    (x1, x2), (y1, y2) = window.view.viewRange()
    state = {
        "version": VERSION,
        "cwd": str(Path.cwd()),
        "geometry": bytes(window.saveGeometry()),
        "docks": bytes(window.saveState()),
        "frames": frames,
        "current": window.current,
        "boxes": {name: getattr(window, name).currentText()
                  for name in ("scale_box", "stretch_box", "cmap_box")},
        "view": (x1, x2, y1, y2) if window.frames else None,
        "examine": dict(window._read_params()),
        "settings": window.examine_settings,
        "measurements": _keepable(window.measurements),
        "logfile": window.examine_logfile,
        "tabs": tabs,
        "tab": window.views.currentIndex(),
        "files": files,
        "file": window.files.tabs.currentIndex(),
        "log": window.session.toPlainText(),
        "prompt": window.prompt.text(),
    }
    data = pickle.dumps(state)
    tmp = here / "state.pickle.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, here / "state.pickle")
    meta = {"version": VERSION, "when": time.strftime("%Y-%m-%d %H:%M"),
            "cwd": state["cwd"],
            "frames": [(f["number"], f["name"]) for f in frames],
            "tabs": titles, "files": [f["path"] for f in files], "unsaved": unsaved}
    tmp = here / "session.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, here / "session.json")
    return here


def _keepable(measurements) -> list:
    out = []
    for item in measurements:
        try:
            pickle.dumps(item)
        except Exception:  # noqa: BLE001
            continue
        out.append(item)
    return out


# ------------------------------------------------------------- restoring


def restore(window, where: Path | None = None) -> list[str]:
    """Put a kept session back into the window. What could not come back, said."""
    from PySide6 import QtCore

    from .lang import trf
    from .window import Frame

    here = where or folder()
    state = pickle.loads((here / "state.pickle").read_bytes())
    notes: list[str] = []
    cwd = Path(state["cwd"])
    if cwd.is_dir():
        os.chdir(cwd)
    else:
        notes.append(trf("the folder {0} is not there any more; staying in {1}",
                         cwd, Path.cwd()))
    window.restoreGeometry(QtCore.QByteArray(state["geometry"]))
    window.restoreState(QtCore.QByteArray(state["docks"]))

    if state["log"]:
        window.session.setPlainText(state["log"])
    window.prompt.setText(state["prompt"])

    for name, value in state["boxes"].items():
        box = getattr(window, name)
        box.blockSignals(True)
        box.setCurrentText(value)
        box.blockSignals(False)
    written = {}
    for entry in state["frames"]:
        try:
            data = np.load(here / entry["file"], allow_pickle=False)
            quality = (np.load(here / f"dq_{entry['number']}.npy", allow_pickle=False)
                       if entry["quality"] else None)
        except OSError:
            notes.append(trf("frame {0} ({1}) could not be read back", entry["number"],
                             entry["name"]))
            continue
        frame = Frame(data, _header_back(entry["header"]), entry["name"],
                      window=entry["window"], table=entry["table"], quality=quality)
        frame.folder = Path(entry["folder"])
        frame.markers = list(entry["markers"])
        window.frames[entry["number"]] = frame
        written[entry["number"]] = (id(frame), entry["file"])
    window._session_frames = written
    if window.frames:
        labels = [f"{n}  {f.name}" for n, f in sorted(window.frames.items())]
        window.frame_box.blockSignals(True)
        window.frame_box.clear()
        window.frame_box.addItems(labels)
        window.frame_box.blockSignals(False)
        current = state["current"] if state["current"] in window.frames else min(
            window.frames)
        window.select_frame(current)

    for name, value in state["examine"].items():
        widget = window.param_widgets.get(name)
        if widget is None:
            continue
        if hasattr(widget, "setCurrentText"):
            widget.setCurrentText(str(value))
        else:
            widget.setText(f"{value:g}" if isinstance(value, float) else str(value))
    window.examine_settings = state["settings"]
    window.measurements = list(state["measurements"])
    window.examine_logfile = state["logfile"]
    if window.measurements:
        window._plot(window.measurements[-1][1])

    for title, kind, payload in state["tabs"]:
        panel = _panel_back(kind, payload)
        if panel is None:
            notes.append(trf("the tab {0} could not be opened again", title))
            continue
        window.add_kept_panel(panel, title)
    if 0 <= state["tab"] < window.views.count():
        window.views.setCurrentIndex(state["tab"])
    else:
        window.views.setCurrentIndex(0)

    for entry in state["files"]:
        path = Path(entry["path"])
        if entry["text"] is None and not path.is_file():
            notes.append(trf("{0} is not there any more", path))
            continue
        if path.is_file():
            window.files.open([str(path)])
        else:
            window.files.open_text(path, "")
        tab = window.files.tabs.currentWidget()
        if tab is None or tab.path != path.absolute():
            continue
        if entry["text"] is not None:
            tab.replace_all(entry["text"])
            tab.document().setModified(True)
        cursor = tab.textCursor()
        cursor.setPosition(min(entry["position"], len(tab.toPlainText())))
        tab.setTextCursor(cursor)
    if 0 <= state["file"] < window.files.tabs.count():
        window.files.tabs.setCurrentIndex(state["file"])
    if window.frames and state["view"] is not None:
        # last, once the window has its size back: the image keeps its aspect,
        # so the same view in another size would be another view
        from PySide6 import QtWidgets

        QtWidgets.QApplication.processEvents()
        x1, x2, y1, y2 = state["view"]
        window.view.setRange(xRange=(x1, x2), yRange=(y1, y2), padding=0)
    return notes


def _panel_back(kind: str, payload):
    from .graph_panel import GraphPanel

    try:
        if kind == "graph" or kind == "picture":
            return GraphPanel(payload)
        if kind == "figure":
            from .figure_panel import FigurePanel

            return FigurePanel(payload)
        if kind == "implot":
            from ..core import images as imageio
            from .implot_panel import ImplotPanel

            engine = payload["engine"]
            data = engine.get("data")
            panel = ImplotPanel(np.zeros((2, 2)) if data is None else data,
                                payload["name"])
            panel.engine.__dict__.update(engine)
            if engine.get("_files"):
                panel.engine.read = lambda name: imageio.read(name)
            else:
                frozen = np.asarray(data, float)
                panel.engine.read = lambda _: (frozen, None)
            panel._render()
            return panel
    except Exception:  # noqa: BLE001 -- said by the caller
        return None
    return None


# ------------------------------------------------------------- at the start


def offer(window, answer: str | None = None) -> str:
    """harwood gui: when a session was kept, ask whether to restore it or start a
    new one; then keep this one. `answer` ("restore" or "new") answers for the
    person (tests). What was chosen ("restore", "new", or "none" with nothing kept)."""
    from PySide6 import QtWidgets

    from .lang import tr

    meta = summary()
    choice = "none"
    if worth_restoring(meta):
        if answer is None:
            box = QtWidgets.QMessageBox(window)
            box.setWindowTitle(tr("Previous session"))
            box.setIcon(QtWidgets.QMessageBox.Icon.Question)
            box.setText(tr("Restore the session harwood had open when it closed?"))
            box.setInformativeText("\n".join(describe(meta)))
            restore_button = box.addButton(tr("Restore previous session"),
                                           QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            box.addButton(tr("Start a new session"),
                          QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(restore_button)
            box.exec()
            answer = "restore" if box.clickedButton() is restore_button else "new"
        choice = answer
    if choice == "restore":
        try:
            window.restore_from()
        except Exception as error:  # noqa: BLE001 -- a broken session is not a broken window
            window.log(tr("  the previous session could not be restored") + f": {error}")
            set_aside()
    elif choice == "new":
        set_aside()
        if meta and meta.get("unsaved"):
            from .lang import trf

            window.log(trf("Text not saved in the previous session ({0}) is kept with "
                           "it: File > Restore previous session brings it back.",
                           ", ".join(Path(p).name for p in meta["unsaved"])))
    window.begin_session()
    return choice
