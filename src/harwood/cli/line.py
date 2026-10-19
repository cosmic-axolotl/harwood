"""One line, several commands -- and help typing them.

    hw> imcombine @bias.lis Zero.fits && ccdproc @obj.lis --zero Zero.fits; list

`;` separates commands, as it does at the cl prompt: the next one runs
whatever happened to the one before. `&&` runs the next only if the one before
worked, and `||` only if it failed -- the shell's meaning, because the shell
is where everyone met them. A command that is skipped says so and says why;
a chain that silently did half of what was typed is worse than one that
stopped.

The splitting knows about quotes, so a `;` inside 'a quoted value' is a
character and not a separator. Each command is then read exactly as a line
with one command always was.

Completion is click's own (the same machinery as bash completion, so the
two cannot disagree about what a task accepts), plus the two things click
leaves to the shell: file names, and imexamine's colon commands. Nothing
here knows about Qt; the window and the terminal session both use it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from ..core import advice

#: How a command is joined to the one before it.
CONNECTORS = (";", "&&", "||")

#: Commands whose arguments are never file names, so an empty completion stays
#: empty instead of filling up with the directory.
NO_FILES = {"help", "lpar", "list", "history"}

#: Characters a completed word has to escape to stay one word.
_SPECIAL = set(" \t;&|'\"\\")


@dataclass(frozen=True)
class Step:
    connector: str      #: "" for the first command, else one of CONNECTORS
    command: str


def split(line: str) -> list[Step]:
    """The commands on a line, each with what joins it to the one before.

    Raises ValueError, with a fix, for a line that cannot mean anything: a
    connector with nothing on one side, or `&` and `|`, which mean background
    jobs and pipes in the cl and the shell and are not something harwood does.
    """
    pieces, _ = _pieces(line)
    texts = [line[a:b].strip() for _, a, b in pieces[0::2]]
    connectors = [kind for kind, _, _ in pieces[1::2]]

    for connector in connectors:
        if connector == "&":
            advice.fail("a single '&' runs a command in the background, in the cl and "
                        "in the shell; harwood runs one command at a time",
                        "write '&&' to run the next command only if this one worked, "
                        "or ';' to run it either way")
        if connector == "|":
            advice.fail("harwood does not pipe one command into the next",
                        "write '&&' or ';' between commands; to filter what a task "
                        "prints, run it from the shell: harwood ... | grep ...")

    steps = []
    for index, text in enumerate(texts):
        before = connectors[index - 1] if index else ""
        after = connectors[index] if index < len(connectors) else ""
        if not text:
            if after in ("&&", "||"):
                advice.fail(f"nothing before '{after}'",
                            f"put a command in front of '{after}', or drop it")
            if before in ("&&", "||"):
                advice.fail(f"nothing after '{before}'",
                            f"add the command to run after '{before}', or drop it")
            continue                    # `a ;; b` and a trailing `;` are harmless
        steps.append(Step(before if steps else "", text))
    return steps


def run(line: str, run_one: Callable[[str], bool],
        report: Callable[[str], None]) -> bool:
    """Run every command on a line; True when the last one that ran worked.

    `run_one` runs a single command and says whether it worked. `report` is
    told about each command that is skipped, and why. A skipped command leaves
    the status as it was, as in the shell: in `a && b; c`, c runs either way.
    """
    ok = True
    for step in split(line):
        if step.connector == "&&" and not ok:
            report(f"skipped: {step.command} -- the command before it failed")
            continue
        if step.connector == "||" and ok:
            report(f"skipped: {step.command} -- the command before it worked")
            continue
        ok = run_one(step.command)
    return ok


# ------------------------------------------------------------- completion


@dataclass(frozen=True)
class Item:
    value: str          #: what replaces the word being typed
    help: str = ""
    kind: str = "plain"  #: plain, file or dir


@dataclass(frozen=True)
class Completion:
    start: int          #: where in the text the word being completed begins
    items: list[Item]

    def common(self) -> str:
        """The longest start every candidate shares."""
        return os.path.commonprefix([item.value for item in self.items])


def complete(text: str) -> Completion:
    """What could come next, for `text` typed up to the cursor.

    Never raises: a half-typed line is the normal case here, and completion
    that fails must leave the line exactly as it was.
    """
    pieces, _ = _pieces(text)
    segment_start = pieces[-1][1]
    words, trailing_space = _words(text[segment_start:])
    if words and not trailing_space:
        start = segment_start + words[-1][0]
        incomplete, done = words[-1][1], [value for _, value in words[:-1]]
    else:
        start, incomplete, done = len(text), "", [value for _, value in words]
    if done[:1] == ["harwood"]:
        done = done[1:]

    try:
        if not done and incomplete.startswith(":"):
            items = _colon(incomplete)
        else:
            items = _click(done, incomplete)
    except Exception:                   # completion must never cost the line
        items = []
    return Completion(start, items)


def _click(done: list[str], incomplete: str) -> list[Item]:
    from click.shell_completion import ShellComplete

    from .main import app

    found = ShellComplete(app, {}, "harwood", "_HARWOOD_COMPLETE").get_completions(
        done, incomplete)
    items: list[Item] = []
    for each in found:
        if each.type in ("file", "dir"):
            items += _files(incomplete, dirs_only=each.type == "dir")
        else:
            items.append(Item(_escape(str(each.value)), each.help or ""))
    if not found and done and done[0] not in NO_FILES and not incomplete.startswith("-"):
        items = _files(incomplete)
    return _one_spelling(items, incomplete)


def _one_spelling(items: list[Item], incomplete: str) -> list[Item]:
    """`--low-reject` and `--low_reject` are one option (build.option_names);
    offer it once, in the spelling being typed."""
    underscore = "_" in incomplete
    seen: dict[str, Item] = {}
    for item in items:
        if not item.value.startswith("--"):
            seen[item.value] = item
            continue
        key = item.value.replace("_", "-")
        if key not in seen or ("_" in item.value) == underscore:
            seen[key] = item
    return list(seen.values())


def _colon(incomplete: str) -> list[Item]:
    from ..core import examine

    word = incomplete[1:]
    refused = examine._STYLING | set(examine._ELSEWHERE)
    return [Item(f":{name}", "imexamine") for name in examine.COMMANDS
            if name.startswith(word) and name not in refused]


def _files(incomplete: str, dirs_only: bool = False) -> list[Item]:
    """Names in the directory being typed. An @ in front -- an IRAF list --
    is kept, and completes the file the list is in."""
    prefix = "@" if incomplete.startswith("@") else ""
    word = incomplete[len(prefix):]
    directory, partial = os.path.split(os.path.expanduser(word))
    typed_directory = word[:len(word) - len(partial)]
    try:
        names = sorted(os.listdir(directory or "."))
    except OSError:
        return []
    items = []
    for name in names:
        if not name.startswith(partial) or (name.startswith(".") and not partial):
            continue
        is_dir = os.path.isdir(os.path.join(directory or ".", name))
        if dirs_only and not is_dir:
            continue
        value = prefix + _escape(typed_directory + name) + ("/" if is_dir else "")
        items.append(Item(value, kind="dir" if is_dir else "file"))
    return items


def _escape(word: str) -> str:
    return "".join("\\" + ch if ch in _SPECIAL else ch for ch in word)


# ------------------------------------------------------------- reading


def _pieces(text: str) -> tuple[list[tuple[str, int, int]], str | None]:
    """The line cut at its connectors, outside quotes: alternately
    ("text", start, end) and (connector, start, end), beginning and ending
    with text. Also the quote still open at the end, if any."""
    pieces: list[tuple[str, int, int]] = []
    quote: str | None = None
    start = i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(text):
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch in "'\"":
            quote = ch
            i += 1
            continue
        connector = next((c for c in ("&&", "||", ";", "&", "|")
                          if text.startswith(c, i)), None)
        if connector is None:
            i += 1
            continue
        pieces += [("text", start, i), (connector, i, i + len(connector))]
        i += len(connector)
        start = i
    pieces.append(("text", start, len(text)))
    return pieces, quote


def _words(text: str) -> tuple[list[tuple[int, str]], bool]:
    """The words of one command as (start, value), quotes taken off -- also
    from a quote still open, which is what a word being typed often has --
    and whether the text ends in a space outside quotes."""
    words: list[tuple[int, str]] = []
    i = end = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        start, value, quote = i, [], None
        while i < len(text) and (quote or not text[i].isspace()):
            ch = text[i]
            if quote:
                if ch == quote:
                    quote = None
                elif ch == "\\" and quote == '"' and i + 1 < len(text):
                    i += 1
                    value.append(text[i])
                else:
                    value.append(ch)
            elif ch in "'\"":
                quote = ch
            elif ch == "\\" and i + 1 < len(text):
                i += 1
                value.append(text[i])
            else:
                value.append(ch)
            i += 1
        words.append((start, "".join(value)))
        end = i
    return words, bool(text) and end < len(text)
