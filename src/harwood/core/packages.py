"""The package tree: IRAF's packages, in IRAF's nesting, with harwood's tasks.

What the click-only tab browses, and what anything else that wants the
structure (`harwood list`, the documentation) can ask for. Built from the
migration map -- which knows every IRAF task, its package and whether
harwood has it -- and data/packages.yaml, which gives each package a brief
line. No Qt here.

    tree()                 -> the root Node
    find(path)             -> the Node at "noao.digiphot.apphot"
    node.path, node.name, node.brief, node.icon
    node.children          -> sub-packages
    node.tasks             -> TaskEntry: name, available, brief, status, note

IRAF writes some package paths short ("digiphot.apphot"); they are made
whole ("noao.digiphot.apphot") so that the tree is the one an IRAF user has
in their head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

#: Packages that live under noao in IRAF, though the inventory names them
#: without it.
UNDER_NOAO = ("imred", "digiphot", "onedspec", "twodspec", "astutil", "obsutil", "rv",
              "artdata", "astcat", "nproto", "mtlocal", "nobsolete")


@dataclass
class TaskEntry:
    """One task in a package."""

    name: str                    #: what to type (harwood's name)
    iraf: str                    #: IRAF's name, when it differs
    available: bool              #: harwood has it
    brief: str
    status: str                  #: available, partial, planned, replaced, wontport
    note: str = ""


@dataclass
class Node:
    """One package."""

    path: str                    #: "" is the root
    brief: str = ""
    icon: str = "·"
    children: list[Node] = field(default_factory=list)
    tasks: list[TaskEntry] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.rsplit(".", 1)[-1] if self.path else "all"

    @property
    def parent_path(self) -> str | None:
        if not self.path:
            return None
        return self.path.rsplit(".", 1)[0] if "." in self.path else ""

    def available_count(self) -> int:
        """Tasks harwood has, here and below."""
        return (sum(t.available for t in self.tasks)
                + sum(c.available_count() for c in self.children))

    def total_count(self) -> int:
        return len(self.tasks) + sum(c.total_count() for c in self.children)


def full_path(package: str) -> str:
    """'digiphot.apphot' -> 'noao.digiphot.apphot'; 'tv' -> 'images.tv'."""
    if package == "tv":
        return "images.tv"
    head = package.split(".", 1)[0]
    if head in UNDER_NOAO:
        return f"noao.{package}"
    return package


@lru_cache(maxsize=1)
def _briefs() -> dict[str, dict]:
    import yaml

    path = Path(__file__).resolve().parent.parent / "data" / "packages.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def tree() -> Node:
    """The whole tree, fresh (cheap enough; the registry is cached)."""
    from . import migration, registry

    briefs = _briefs()
    nodes: dict[str, Node] = {}

    def node(path: str) -> Node:
        if path not in nodes:
            info = briefs.get(path, {})
            nodes[path] = Node(path, " ".join(str(info.get("brief", "")).split()),
                               info.get("icon", "·"))
            if path:
                parent = node(path.rsplit(".", 1)[0] if "." in path else "")
                parent.children.append(nodes[path])
        return nodes[path]

    node("")
    available = registry.list_tasks()
    entries = {**migration.tasks(), **migration.added()}
    for iraf_name, entry in entries.items():
        name = entry.get("harwood", iraf_name)
        packages = entry["pkg"] if isinstance(entry["pkg"], list) else [entry["pkg"]]
        has = name in available
        if has:
            brief = available[name].description
        else:
            brief = entry.get("note", "") or "In IRAF; not in harwood."
        note = " ".join(str(entry.get("note", "")).split())
        for package in packages:
            node(full_path(package)).tasks.append(TaskEntry(
                name=name, iraf=iraf_name, available=has,
                brief=" ".join(brief.split()), status=entry.get("status", "planned"),
                note=note))
    for each in nodes.values():
        each.children.sort(key=lambda n: (n.available_count() == 0, n.name))
        each.tasks.sort(key=lambda t: (not t.available, t.name))
    return nodes[""]


def find(path: str, root: Node | None = None) -> Node:
    """The node at a dotted path; KeyError, with the nearest names, if none."""
    root = root or tree()
    if not path:
        return root
    current = root
    walked = []
    for part in path.split("."):
        walked.append(part)
        match = next((c for c in current.children if c.name == part), None)
        if match is None:
            names = ", ".join(c.name for c in current.children)
            raise KeyError(f"no package '{'.'.join(walked)}'; here there are: {names}")
        current = match
    return current


def named(name: str, root: Node | None = None) -> Node | None:
    """The package called `digiphot` or `noao.digiphot`, wherever it is; None
    if IRAF has no such package."""
    if not name:
        return None

    def walk(node: Node) -> Node | None:
        for child in node.children:
            if name in (child.name, child.path):
                return child
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(root or tree())


def names_under(node: Node) -> set[str]:
    """Every task of a package and of the packages inside it."""
    names = {task.name for task in node.tasks}
    for child in node.children:
        names |= names_under(child)
    return names


def locate(task: str, root: Node | None = None) -> list[str]:
    """Every package path a task appears in."""
    root = root or tree()
    found = []

    def walk(node: Node) -> None:
        if any(t.name == task or t.iraf == task for t in node.tasks):
            found.append(node.path)
        for child in node.children:
            walk(child)

    walk(root)
    return found
