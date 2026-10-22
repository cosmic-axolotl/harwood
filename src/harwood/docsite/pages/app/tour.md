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
