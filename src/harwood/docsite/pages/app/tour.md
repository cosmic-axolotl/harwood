# The window

`harwood gui` opens one window with everything in it: the image in the
middle, and around it panels you can move, close and bring back.

| panel | where it starts | what it is for |
|---|---|---|
| **Tasks** | left | every task by clicks: IRAF's package tree, a form per task ([more](page:app/clicks)) |
| **Examine** | right | what `imexamine` measured: the radial profile and its fit |
| **Parameters · rimexam** | right | imexamine's settings, each one explained |
| **Command line** | bottom | what ran and what it said, and the `hw>` prompt ([more](page:app/command-line)) |
| **Files** | bottom, when opened | text files in tabs, and editing several at once ([more](page:app/files)) |
| **Documentation** | bottom, when opened | these pages (F1) |

## Arranging it

Every panel can be dragged by its title to another side of the window, or
out of it to float on its own. Drop one panel onto another and the two
become tabs of the same place. Closing a panel does not lose it: the
**Panels** menu lists them all, and **Put the panels back**
(Ctrl+Shift+R) returns every one to where it started.

| keys | |
|---|---|
| Ctrl+T | show or hide Tasks |
| Ctrl+L | go to the prompt |
| Ctrl+O | open text files |
| F1 | the documentation |
| F11 | full screen |
| Ctrl+M | maximise |
| Ctrl+ and Ctrl- | larger and smaller text; Ctrl+0 back to normal |

## How it looks

The **View** menu has the choices, and they apply to everything at once,
the plots and these pages included:

- **Dark**, **Light**, or **Follow the system**.
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

The choices are kept in `~/.config/harwood/interface.json`, a file you can
read; *View › Where these choices are kept* shows it. They are not reduction
parameters, so they live with you and not with the night.

## Saving what you see

The **File** menu exports the image as it is displayed (PNG, JPEG, TIFF),
the view with its marks, and any plot (PNG, SVG, PDF). From the command
line, [impicture](task:impicture) does the same for any image.

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
