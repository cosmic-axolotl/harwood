IRAF's `precess`: a list of coordinates from one equinox to another, printed as IRAF
prints them. Each line of a file has `ra dec`, in hours and degrees, decimal or sexagesimal;
it may go on with the equinox of that line and the year to take it to, where
`--startyear` and `--endyear` do not apply. `STDIN` reads the lines from the terminal.

`--stdepoch`, if not 0, prints each a second time for that year.

A line that does not begin with two numbers is passed over, and the command says so.
The lines are what IRAF 2.18 prints, in a real `RADIAN` for its conversions, which is where
the last digit comes from.
