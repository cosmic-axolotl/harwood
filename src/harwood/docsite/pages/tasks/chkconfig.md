Compiles a photcal configuration file as [fitparams](task:fitparams) and
[invertfit](task:invertfit) compile it, and says what is wrong with it and
where -- or, with `--verbose`, what it declares: each variable's column
and error column, the parameters and their initial values, the equations
as the parser read them, the step or derivative of each parameter.

```console
$ harwood chkconfig configuration.mkconfig --verbose
```

**Reading the line numbers.** Each message names the line the parser had
reached, and shows it. For most mistakes that is the culprit's own line;
but a statement that ends in an expression is only known to have ended
when the next one starts, so an equation declared twice, or an error
equation for one that does not exist, is reported on the line *after* it.
What is found once the whole file is read -- two variables reading the
same column -- is reported past the last line, with no line shown. A
syntax error stops the compilation, so fix the messages from the first:
one mistake can cause the next.

**Two things that catch people.** Keywords can be shortened to three
letters, so a variable called `err`, `fit` or `obs` is read as a keyword;
and the last line needs its newline -- without one IRAF cannot read the
file at all, which chkconfig reports as a syntax error at `''` there.

**Identical to IRAF.** The output is IRAF's chkconfig's, byte for byte, on
the standard output alone (the "syntax error" yacc adds goes to the
standard error, as in IRAF), checked on 37 configurations: the Skinakas
one, one for each standards catalog as [mkconfig](task:mkconfig) puts it
together, and a set written to make every message the parser has. One
difference: a file ending in `#` with no newline sends IRAF into an
endless loop; harwood reports the syntax error.
