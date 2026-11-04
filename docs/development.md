# Working on harwood: Linux, or Windows through WSL

harwood is developed on Linux. On Windows the same setup runs inside WSL 2
(Ubuntu 24.04 or later), with the window shown on the Windows desktop by WSLg
and the editor -- VS Code, say -- running on Windows.

## Get the code

Clone **inside WSL's own file system** (`~/...`), with WSL's git, not under
`/mnt/c`: many tests compare files byte for byte with IRAF's, and a checkout
made by Git for Windows turns their line ends into CRLF. `/mnt/c` is also
several times slower for the thousands of small files the suite reads.

```bash
sudo apt update
sudo apt install -y git python3 python3-venv
cd ~
git clone https://github.com/cosmic-axolotl/harwood.git
cd harwood
git config core.autocrlf false
```

To push, sign in once (`sudo apt install gh && gh auth login`, or Git
Credential Manager from Windows).

## The environment

Python 3.12 or later (Ubuntu 24.04's is 3.12).

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[all,dev]"
.venv/bin/harwood doctor
```

The window needs Qt's system libraries. Under WSLg it is drawn through X11,
which wants these:

```bash
sudo apt install -y libegl1 libgl1 libfontconfig1 libdbus-1-3 \
    libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xkb1
.venv/bin/harwood gui
```

If Qt still says it cannot load the "xcb" platform plugin, run once with
`QT_DEBUG_PLUGINS=1` and install the library it names.

## The tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q --capture=sys
```

`offscreen` keeps the windows of the GUI tests off the screen; `--capture=sys`
leaves the real stderr in place, so that a crash inside Qt says where it was.

Most of the tests compare with output IRAF wrote, frozen in `tests/data/`.
Some also need the IRAF tutorial's frames (M92 and the 1992 spectroscopic
night), which are not in the repository: put the `exercises` directory --
`intro/`, `phot/`, `spec/` ... -- at `~/iraf/exercises`, or point
`HARWOOD_TEST_DATA` at it. Without it those tests are skipped, and say why.

The same suite runs on GitHub Actions on every push, on Linux and on Windows
(`.github/workflows/`), against the installed package rather than the
checkout. A Windows run takes about 25 minutes, so it pays to push several
fixes together.

## IRAF, as the reference

Each port is held to what IRAF itself prints. To make a new reference --
the `run_iraf*.py` script next to each set of frozen output in `tests/data/`
says how it was made -- IRAF 2.18 and PyRAF are needed, in the system Python,
not in the venv:

```bash
sudo apt install -y iraf iraf-noao python3-pyraf
python3 -c "from pyraf import iraf; iraf.noao(); print('ok')"
```

The SPP sources IRAF is ported from are in the iraf-community repository
(`https://github.com/iraf-community/iraf`); port the code, not the help page,
which is wrong in places. Code translated from IRAF keeps the AURA notice
(`LICENSES/IRAF-COPYRIGHT`).

Sessions typed at IRAF's prompts are recorded through a pseudo-terminal
(`pexpect`, `sudo apt install python3-pexpect`), so that Ctrl-D ends one
list and the next can still be typed, as at a keyboard: see
`tests/data/iraf_photcal/mkconfig/run_iraf_typed.py`.

## Where the work is

`harwood list --iraf` lists every IRAF task and what harwood does with it,
from `src/harwood/data/iraf_migration.yaml`; `partial` entries name their
gaps in `note`. The counts at the top of that file are checked by the suite,
so change them with the statuses.
