# The command line

The prompt at the bottom of the window (Ctrl+L goes to it) and the terminal
session (`harwood` on its own) take the same commands as a shell, without
the `harwood` in front -- though it does no harm:

```text
hw> imstatistics red/obj001.fits
hw> help phot
hw> daofind --plan red/obj001.fits
```

## Which directory

`cd` and `pwd` work as in a shell, and in the cl. Tasks read and write in the
working directory, and its `harwood.yaml` and log are the ones in use, so each
directory is a reduction:

```text
hw> cd ~/iraf/skinakas
hw> pwd
hw> imstatistics stan0051.fits
hw> cd -
```

`cd` alone goes home, `cd -` goes back to the one before, and Tab completes
directories. A `!cd` would change only the shell of its own that `!` runs in, so
harwood does the `cd` itself. From a terminal, `harwood cd` refuses: the command
ends at once and the directory would be forgotten, so use your shell's.

## Naming what you write

Name the output, and write an image's extension:

```text
hw> zerocombine bias*.fits Zero.fits
hw> daofind stan0051.fits stan0051.coo --fwhmpsf 6.5
```

The name is used exactly as typed: `Zero` makes a file called `Zero`, which
`*.fits` and `Zero.fits` do not find. With no name at all the result goes to
[a directory for the run](page:concepts/outputs), where the next task does not
look unless told. Both cases work, and a note after the run says so.

## Several commands on a line

| between two commands | the second one runs |
|---|---|
| `;` | whatever happened to the first, as in the cl |
| `&&` | only if the first worked |
| `\|\|` | only if the first failed |

```text
hw> imstatistics red/obj001.fits && daofind red/obj001.fits obj001.coo --fwhmpsf 3; list apphot
```

(In a terminal, outside the session, the shell itself already reads `;`
and `&&` between `harwood` commands.)

A command that is skipped says so and why -- *skipped: daofind … -- the
command before it failed* -- rather than leaving you to notice that half of
the line did nothing. Quotes are respected: an `&&` inside a quoted
`hselect` expression is part of the expression. A single `&` (a background
job) or `|` (a pipe) is refused with what to write instead.

## Tab

Tab completes the word under the cursor: a task name, with what each task
does; an option, in the spelling you started (`--low-reject` or
`--low_reject`); the values an option allows (`--combine` offers average,
median, …); a file, a directory, or an `@list`; imexamine's `:` commands.
One candidate is written in; several are listed, and the part they share is
written in. Tab on an empty prompt moves on to the next control, as
everywhere else, so a keyboard can always leave the prompt.

## History

What you type is kept, in the window and in the terminal alike, in one file
(`~/.local/state/harwood/history`), so the up arrow tomorrow finds what you
typed today in either place.

- ↑ and ↓ walk it. With something typed, only the lines that start with it:
  type `imc` and ↑ finds the last imcombine.
- Ctrl+R, or the **History** button, searches all of it for what is typed,
  anywhere in a line.
- `history` numbers it; `history 50` shows more, `history --find Zero` only
  the lines that mention Zero.

The history is what was typed, typos included. What the tasks *did* is the
[log](page:concepts/outputs), which also says how to repeat each run
exactly.

## One at a time

The window runs one command at a time. Pressing Run or Enter while one is
running says that it is still running and leaves your line at the prompt;
the window will not close under a task that is writing files.
