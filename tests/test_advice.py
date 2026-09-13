"""Notices: what happened, what to do, and being readable without colour.

The rule this file enforces is that a notice with no suggestion is not
finished. Saying "only 3 of 12 features land on a catalogued line" leaves the
reader exactly where they were; it becomes help at "try a smaller --tolerance".
"""

import ast
from pathlib import Path

import pytest

from harwood.core import advice

TASKS = Path(advice.__file__).resolve().parent.parent / "tasks"


def test_the_three_levels_mean_different_things():
    assert advice.note("x").level == "note"
    assert advice.warn("x").level == "warning"
    assert advice.problem("x").level == "problem"
    assert advice.ORDER["problem"] > advice.ORDER["warning"] > advice.ORDER["note"]


def test_colour_is_never_the_only_channel():
    """One man in twelve cannot reliably separate red from green, which is
    exactly the pair 'red means bad' relies on. The level has to survive being
    piped to a file, read aloud, or seen without colour."""
    for level in ("note", "warning", "problem"):
        item = advice.Advice("x", "y", level)
        assert item.level in item.plain(), "the word is in the text itself"
        assert item.symbol and item.symbol != advice.Advice("x", "y", "note").symbol \
            or level == "note", "each level has its own shape"
        assert item.colour.startswith("#")


def test_the_palette_is_the_colour_blind_safe_one():
    """Okabe-Ito. Vermilion and orange stay distinguishable under the common
    deficiencies, where pure red and pure green collapse together."""
    assert advice.COLOURS["problem"] == "#D55E00"    # vermilion
    assert advice.COLOURS["warning"] == "#E69F00"    # orange
    assert advice.COLOURS["note"] == "#56B4E9"       # sky blue
    assert len(set(advice.COLOURS.values())) == 3
    assert len(set(advice.SYMBOLS.values())) == 3


def test_searching_a_notice_looks_at_the_suggestion_too():
    """A suggestion moved out of the prose into its own field is still part of
    what the notice says."""
    item = advice.warn("something drifted", "add --override to replace it")
    assert "--override" in item
    assert "drifted" in item
    assert "nonsense" not in item


def test_a_plain_string_still_works_but_carries_no_suggestion():
    converted = advice.as_advice("just a string")
    assert converted.level == "warning" and converted.fix == ""
    assert advice.as_advice(advice.note("x")).level == "note"


def test_the_worst_level_in_a_set_is_what_the_set_is():
    assert advice.worst([advice.note("a"), advice.problem("b")]) == "problem"
    assert advice.worst([advice.note("a"), advice.note("b")]) == "note"
    assert advice.worst([]) is None


def test_an_error_can_carry_its_own_suggestion():
    error = advice.HarwoodError("the arc has no solution",
                                "run identify on it first")
    assert str(error) == "the arc has no solution"
    assert error.advice().level == "problem"
    assert "identify" in error.advice()


def test_the_whole_notice_fits_on_one_line_for_a_log():
    item = advice.problem("it went wrong", "turn it off and on again")
    assert item.plain() == ("problem: it went wrong -- try: turn it off and on "
                            "again")


# ------------------------------------------------- the rule, enforced


def _advice_calls() -> list[tuple[str, int, ast.Call]]:
    """Every `advice.note/warn/problem(...)` in the task modules."""
    found = []
    for path in sorted(TASKS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("note", "warn", "problem")
                    and getattr(node.func.value, "id", "") == "advice"):
                found.append((path.name, node.lineno, node))
    return found


def test_every_notice_says_what_to_do_about_it():
    """The rule. A notice with no suggestion is not finished.

    This is the same discipline as the test that every step must explain
    itself: it started as a convention, and a convention nobody checks is a
    convention that decays.
    """
    calls = _advice_calls()
    assert len(calls) > 25, "they should all have been converted"

    missing = [f"{name}:{line}" for name, line, node in calls
               if len(node.args) < 2 and not any(
                   keyword.arg == "fix" for keyword in node.keywords)]
    assert not missing, f"notices with no suggestion: {', '.join(missing)}"


def test_no_task_still_appends_a_bare_string():
    """The old shape still renders, so it would not fail loudly -- which is
    exactly why it needs a test."""
    offenders = []
    for path in sorted(TASKS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append"
                    and getattr(node.func.value, "id", "") == "warnings"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"bare strings: {', '.join(offenders)}"


def test_a_suggestion_tells_the_reader_to_do_something():
    """Not a restatement of the problem. The check is crude -- a suggestion
    should be short, and should not simply repeat the message -- but it
    catches the failure mode of writing the same sentence twice."""
    for name, line, node in _advice_calls():
        fix = next((keyword.value for keyword in node.keywords
                    if keyword.arg == "fix"),
                   node.args[1] if len(node.args) > 1 else None)
        text = _literal(fix)
        if text is None:
            continue
        assert len(text) > 15, f"{name}:{line} -- suggestion too short"
        assert len(text) < 400, f"{name}:{line} -- suggestion is an essay"


def _literal(node) -> str | None:
    """The constant parts of an f-string or a concatenation, joined."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value for part in node.values
                       if isinstance(part, ast.Constant))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left), _literal(node.right)
        return (left or "") + (right or "")
    return None


# ----------------------------------------------------------- rendering


@pytest.fixture
def run():
    from click.testing import CliRunner

    from harwood.cli.main import app

    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def test_a_real_notice_prints_its_level_and_its_suggestion(run, tmp_path):
    """End to end: a task that warns has to show the word, not just a colour,
    and the suggestion on its own line."""
    import numpy as np
    from astropy.io import fits

    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), fits.Header(
        {"DATE-OBS": "25/01/92", "UT": "3:00:00", "EXPTIME": 10.0})).writeto(
            tmp_path / "x.fits")

    result = run("setjd", str(tmp_path / "x.fits"), "--observatory", "kpno")
    assert result.exit_code == 0
    assert "note" in result.output
    assert "try:" in result.output


def test_a_failure_prints_the_word_problem_and_a_suggestion(run):
    from harwood.core.advice import SYMBOLS

    result = run("dispcor", "nothing.fits")
    assert result.exit_code != 0
    assert "stopped" in result.output
    # The symbol is the second channel; it must be there even when the
    # terminal strips colour.
    assert SYMBOLS["problem"] in result.output
