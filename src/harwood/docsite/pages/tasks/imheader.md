`imheader` reads the headers, not the pixels, so it is quick over a whole
night. Without options it prints one line per image -- size, image type,
exposure, filter, object -- the keywords a reduction turns on, because they
decide which frame is which. The filter is looked for as `FILTER`,
`FILTERS`, `FILTER1` or `INSFILTE`, since observatories disagree about its
name.

`--keywords` prints just the ones asked for, in that order; `--longheader`
prints every card.

```example
title: One line per frame
run: harwood imheader @obj.lis
output: |
  obj001.fits │ 120x100 │ object │ 60
```

```example
title: The keywords you need, in your order
run: harwood imheader @obj.lis --keywords UT,EXPTIME,GAIN
output: |
  02:05:00
```

To pick frames *by* their headers, use [hselect](task:hselect); to read one
keyword into a script, [imgets](task:imgets).
