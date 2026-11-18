# The window

`harwood gui` opens one window with everything in it: the image in the
middle, and around it panels you can move, close and bring back.

| panel | where it starts | what it is for |
|---|---|---|
| **Tasks** | left | every task by clicks: IRAF's package tree, a form per task ([more](page:app/clicks)) |
| **Examine** | right | what `imexamine` measured: the radial profile and its fit |
| **Parameters · rimexam** | right | imexamine's settings, each one explained |
| **Keys** | under the image | every key that works on the image now, and the DQ legend (F2) |
| **Command line** | bottom | what ran and what it said, and the `hw>` prompt ([more](page:app/command-line)) |
| **Files** | bottom, when opened | text files in tabs, and editing several at once ([more](page:app/files)) |
| **Documentation** | bottom, when opened | these pages (F1) |

## The keys, always in view

Whatever is on the screen, the keys that work on it are listed under it, in
as many columns as the window is wide: lower case belongs to the task, upper
case and `:` to the window, as in IRAF's cursor mode. While an image is
merely shown they are imexamine's (`r` a radial profile, `a` photometry...);
while [tvmark](task:tvmark) or phot's cursor mode holds the cursor they are
that task's, and they come back when it ends. The panels that take keys of
their own -- apedit, identify, icfit, implot, standard, sensfunc -- list all
of theirs along their foot. The list is read from the same tables that
answer the keys, so a key cannot be listed that does nothing. Hover over it
for each key's full description; `?` writes them to the session log.

## Graphs are tabs

A graph opens as a tab beside the image, in the middle of the window, not in a
little window of its own, and each new one is a new tab: `implot`, `graph`, the
curve of growth of [mkapfile](task:mkapfile) and the fits of
[fitparams](task:fitparams) stay after `q`, to go back to, until their × closes
them. A panel that is a step in a task (icfit,
identify, apedit...) closes when it ends and goes back to where you were.
The tabs appear with the second view; **Ctrl+PgDown** and **Ctrl+PgUp** walk
through them, and the Keys panel gives its room to the graph, whose own keys
are at its foot. Closing a panel with a task still waiting on it is the same
as `q`. Run from a shell, with no window open, a panel is a window of its own.

## The DQ on the image

An image with a DQ plane (design contract, section 12) shows its flagged
pixels over it, each reason in its own colour -- saturated, a cosmic ray, a
bad column -- and the Keys panel counts them by reason. Colour is never all
there is: under the cursor the status bar names every reason the pixel has
(`DQ = saturated, cosmic ray`). **D** hides the layer and shows it again.

## Arranging it

Every panel can be dragged by its title to another side of the window, or
out of it to float on its own. Drop one panel onto another and the two
become tabs of the same place. Closing a panel does not lose it: the
**Panels** menu lists them all, and **Put the panels back**
(Ctrl+Shift+R) returns every one to where it started.

| keys | |
|---|---|
| Ctrl+T | show or hide Tasks |
| F2 | show or hide Keys |
| Ctrl+L | go to the prompt |
| Ctrl+O | open text files |
| F1 | the documentation |
| F11 | full screen |
| Ctrl+M | maximise |
| Ctrl+ and Ctrl- | larger and smaller text; Ctrl+0 back to normal |

## How it looks

The **View** menu has the choices, and they apply to everything at once,
the plots and these pages included:

- **Dark**, **Light**, **Follow the system**, or **By the hour: dark at
  night** -- dark from 18:00 to 06:00 unless *Night hours...* sets other
  times; the window turns by itself at dusk and at dawn.
- **Colour-blind safe colours**: the data and the warnings take Okabe and
  Ito's palette, which stays distinguishable in the common colour-vision
  deficiencies. Colour is never the only sign anyway: every warning has a
  word and a symbol, and rejected points change shape as well as colour.
- **Dyslexia-friendly text**: Atkinson Hyperlegible (or OpenDyslexic, under
  *Dyslexia font*), wider letter and line spacing, and a cream ground in the
  light theme.
- **Text follows the window size**: text grows with the window, on top of
  Ctrl+ and Ctrl-.
- **Reduce motion**: no animations.

harwood's icon -- a lowercase h whose arch is the Sun's spectrum, the Planck
curve of 5777 K, with a star over its peak -- is drawn in the theme's colours
and changes with it. On Linux, `harwood shortcut` puts harwood in the
applications menu, to pin to the dock, its icon in the same colours (GNOME
shows a new one the next time harwood opens); `harwood shortcut --remove`
takes it away. On Windows and macOS the window shows the icon by itself.

The choices are kept in `~/.config/harwood/interface.json`, a file you can
read; *View › Where these choices are kept* shows it. They are not reduction
parameters, so they live with you and not with the night.

## The cursor

Under the tabs, a strip of boxes chooses the cursor's helpers, kept for next
time: **Crosshair cursor** (a horizontal and a vertical line across the plot,
crossing at the mouse, as IRAF's cursor), **Values on the axes** (with it: the
values where its lines meet the plot's edges, the pixel's x and y on the
image), and, on the image only, **Aperture** (imexamine's circles: the radius
and the sky annulus, from the Parameters panel), **Pixel** (the square of the
pixel under the mouse) and **Centre** (where `r` would centre the star). They
are drawn in a colour no curve or mark uses, and never change what a plot shows.

## The session

When `harwood gui` closes, it keeps what it had open: the folder it was in, the
window and its panels, each frame (its pixels, header, z1 z2, marks) with the
scale, stretch, colours and zoom, imexamine's parameters and measurements, the
graph tabs (implot alive again; a fit whose task is over, as the picture it
was), the text files -- with what was typed and not yet saved -- and the
command line's record. It also keeps it every two minutes, against a crash.

The next time, it asks: **Restore previous session** or **Start a new session**.
A new one sets the old aside, and *File › Restore previous session* still
brings it back; *File › New session* starts afresh at any time. The session
lives in `~/.local/state/harwood/session`, with a `session.json` you can read.
A window that a task opens from the terminal keeps no session.

## Frames

Beside the frame box in the toolbar, ◀ and ▶ go to the frame before and after
(`p` and `n` over the image), and 🗑 takes the frame off the display -- the one
beside it is shown, and the file itself is not touched.

## Saving what you see

**💾 Save**, at the right of the strip under the tabs (or Ctrl+S), saves what is
in view: on the image, the image as displayed (PNG, JPEG, TIFF) or the view with
its marks; on a graph or a spectrum, the graph (PNG, SVG, PDF, or its numbers as
CSV). The dialog is the desktop's own; what was written is said in the strip and
goes by itself. A graph's right-click *Save...* is the same. The **File** menu
has the same exports; from the command line, [impicture](task:impicture) does it
for any image.

## The command line

A long command wraps and the box grows with it (up to six lines), so the whole
of it stays in view; Enter runs it, and ↑ ↓ move through its lines before they
reach the history.

## Time

A task runs beside the window, which keeps answering while it works: what
the task prints appears as it is printed, and a line above the prompt
counts the time -- with an estimate of how long it has left, once the task
has run on this computer before:

```text
⏳ ccdproc · 0:12 · about 0:28 left (from 5 earlier runs on this computer)
```

The estimate comes from this computer's own record of earlier runs of the
same task, scaled by the number of pixels it reads. A task that has never
run here says *no estimate yet* rather than guessing. While a task waits for
you in a panel -- an interactive fit, say -- the line says so instead of
counting.
