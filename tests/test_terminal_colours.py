"""A console opened in a terminal decides then that it colours what it prints; where the
output goes somewhere else -- the window's session, a file it is redirected to -- the
colours must not follow as escape codes (they showed as boxes and [2m in the window)."""

from __future__ import annotations

import io

from rich.console import Console

from harwood.cli import line as command_line


def test_colours_are_taken_out_and_the_text_kept():
    coloured = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor",
                       width=60)
    coloured.print("[green]ok[/green] imstatistics in [bold cyan]0.00s[/]")
    coloured.print("[link=https://example.org]a link[/link]")
    text = coloured.file.getvalue()
    assert "\x1b" in text
    plain = command_line.strip_ansi(text)
    assert "\x1b" not in plain
    assert plain.splitlines() == ["ok imstatistics in 0.00s", "a link"]


def test_a_file_written_through_plain_has_no_colours():
    sink = io.StringIO()
    out = command_line.Plain(sink)
    assert out.write("\x1b[2mimage\x1b[0m  mean\n") == len("\x1b[2mimage\x1b[0m  mean\n")
    assert sink.getvalue() == "image  mean\n"
    assert not out.isatty()


def test_the_window_s_session_shows_no_colours():
    from harwood.viewer.window import _Stream

    stream = _Stream()
    stream.write("\x1b[32mok\x1b[0m imstatistics\n\x1b[2m image \x1b[0m")
    assert stream.take(final=True) == ["ok imstatistics", " image "]
    assert "\x1b" not in stream.everything()


def test_a_redirected_session_command_writes_plain_text(tmp_path, monkeypatch):
    """`imheader x --longheader > hdr` in a terminal session: the file has the header,
    not the console's colours."""
    from harwood.cli import main, session

    monkeypatch.chdir(tmp_path)
    import numpy as np
    from astropy.io import fits

    header = fits.Header()
    header["OBJECT"] = "colours"
    fits.writeto(tmp_path / "x.fits", np.zeros((2, 3), np.float32), header)
    monkeypatch.setattr(main, "console", Console(force_terminal=True,
                                                 color_system="truecolor"))
    assert session._run("imstatistics x.fits > stats")
    text = (tmp_path / "stats").read_text(encoding="utf-8")
    assert "x.fits" in text and "\x1b" not in text
