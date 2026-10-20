Sets, adds or deletes header keywords, on one image or a whole set at once
-- a keyword edit is nearly always wanted on all the flats, or all of a
night, and doing it a file at a time is how half a set ends up different
from the other half.

```example
title: Write the observer into every object frame
run: harwood hedit @obj.lis OBSERVER --value "A. Observer" --add
output: |
  added
```

- Without `--add`, a keyword that is not there is reported and left alone:
  a typo in a keyword name would otherwise create a new keyword and leave
  the real one as it was.
- `--delete` removes the keyword instead.
- `--no-update` shows what would change and changes nothing.
- A value that looks like a number becomes a number, and `yes`/`no` become
  booleans, unless quoted.

What each keyword said before goes into the header's HISTORY: headers are
edited precisely when something was wrong, which is when the old value
matters.
