# By clicks: the Tasks panel

The Tasks panel is IRAF's package tree, walked one level at a time, as you
would at the cl prompt typing `noao`, `digiphot`, `apphot` -- with a line
or two at each level saying what is in it:

```text
all › noao › digiphot › apphot › daofind
```

Two things IRAF never had:

- **A way back that is always on screen.** The *◀ Back to …* button says
  where it goes, and every part of the path above it is a link. Alt+←, Esc
  and Backspace go back too. There is nothing to unload.
- **Search.** Type part of a name -- `daofi` -- and Enter opens it, wherever
  it lives.

*Also show IRAF tasks harwood does not have yet* lists the rest of each
package, greyed, each with what became of it: planned, replaced by
something else, or not ported and why.

## The form

A task opens as a form made from its parameters:

- the parameters it cannot run without come first; the rest are under
  **More options**;
- a file parameter has **Choose…**, which takes several files where the
  task takes a list;
- a value that will not do is marked as you type it, with a dashed border
  and a sentence saying why -- never by colour alone;
- a task that reads an image starts with the one on display;
- **📖 Manual** opens the task's page of this documentation.

Underneath, **The same, typed** shows the command the form amounts to,
changing as you fill it in. **Copy** puts it on the clipboard, for a
terminal or a script; **Edit as text** takes it to the prompt.

**Plan** and **Run** run exactly that command, through the same prompt as
typing it: it appears in the Command line panel as though it had been typed,
and goes into the log and the history the same way. Clicking and typing are
not two ways of doing a thing that might come out differently; they are one
way, and the form shows you the typing.

Every example in this documentation has a **Clicks** tab with the steps for
the form.
