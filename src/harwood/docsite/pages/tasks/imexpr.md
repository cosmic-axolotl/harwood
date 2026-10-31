IRAF's `imexpr`: an image from an expression over images, numbers and header
keywords. The letters `a` to `z` are the operands, given as options; `I`, `J`, `K`
are the coordinates of the pixel:

```example
title: Subtract the median of each line
after: reduced
run: harwood imexpr "a - median(a)" flat.fits --a red/obj001.fits
output: |
  -> flat.fits
```

```example
title: Counts per second, with the exposure from the header
after: reduced
run: harwood imexpr "a / a.EXPTIME" rate.fits --a red/obj001.fits
output: |
  -> rate.fits
```

```example
title: A circular mask, with no image at all
run: harwood imexpr "sqrt((I-b)**2 + (J-c)**2) <= d" mask.fits --b 256 --c 256 --d 100 --dims 512,512
output: |
  -> mask.fits
```

An operand is an image or section, a number, or `b.KEYWORD`, a keyword of operand
`b`; `a.EXPTIME` can also go straight in the expression. The expression can be
`@file`, with its line ends as spaces.

The expression runs once for each line of the output, and that decides how images
of different shapes go together. An image with fewer lines starts over, so a line
or a plane is applied to a whole frame (`a - b` with `--b "a.fits[*,10]"`). Two
lines of different lengths give the shorter. A function of a vector, like `median`,
is a function of one line. `--dims` gives the axes when no image does, and the first
line evaluated decides any length not given: without `--dims`, `I` alone is 32768
long, as in IRAF.

### Types

The numbers are those IRAF's types give, which is not always arithmetic:

- An image is read in its own type, so `short * short` wraps at 16 bits, and
  `int / int` truncates. `--intype real` reads everything as real at least.
- A binary operation runs in the wider of its two types: short, int, long, real,
  double. An integer constant is an int, one with a point a real, and with more
  than ten characters a double.
- A real is single precision, and a function of a real is the double function
  of it rounded to a real.
- The output's pixels are the expression's type, a comparison an int, unless
  `--outtype` says otherwise; going to a smaller type truncates and wraps. `ref`
  is the type of the image whose header it gets (`--refim`, by default the first
  image of most dimensions). A long is stored in 32 bits.

### What can be written

| | |
|---|---|
| `+ - * / **` | arithmetic. `**` is left associative, and `-a ** 2` is `-(a ** 2)` |
| `//` | joins vectors, or strings |
| `< <= > >= == !=` | comparisons, of strings too; `?=` a string against a pattern |
| `&& \|\|` | one level, and only of integers or comparisons: not of reals |
| `& \| ^ ~` | bits |
| `!` | not, of integers and comparisons |
| `c ? x : y` | pixel by pixel; the last arm takes all there is to the right |
| `@"name"` | a keyword whose name has odd characters, as `@"a.DATE-OBS"` |

| | |
|---|---|
| `abs max min mod sqrt` | |
| `sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10` | in radians |
| `deg rad` | radians to degrees and back |
| `int nint short long real double bool str` | conversions; `int` truncates |
| `len hiv lov sum median` | a number from a line |
| `mean stddev` | a number from a line, with rejection: `mean(a, 3)` at 3 sigma, and with no second argument Chauvenet's relation. They are not plain averages |
| `median(a, b, c)` | 3 to 5 vectors, pixel by pixel. With 4 it is the lower middle |
| `repl(a, n)` | `a` over and over |
| `sort(a)`, `shift(a, n)` | of a line; a shift wraps around |

`--rangecheck`, the default, makes a division by zero and the logarithm or root of
what has none 0, so that one bad pixel does not stop the rest; a real divisor under
1e-20 counts as 0. A conditional can deal with those pixels better.

### Macros

`--exprdb` names a file of macros, as IRAF's help has it:

```
MYPI=       3.1415926535897932385
div10(a)    ((a) / 10)
divz(a,b)   ((abs(b) < .000001) ? 0 : a / b)
```

A macro goes on over the lines that start with a blank, and `#` starts a comment.

### A section of the output

`harwood imexpr "a * 2" "b.fits[10:20,10:20]" --a c.fits` writes into a section of
an image that exists, in that image's type, and leaves the rest of it as it was.

### Against IRAF

In 591 expressions the bits of each pixel, the BITPIX, the shape and the keywords
are what IRAF 2.18 writes, and what it refuses is refused here. On purpose, this
much is not the same:

- A NaN stays a NaN. IRAF reads it as 0 and goes on.
- Where IRAF dies there is an answer: images of 64-bit integers ("integer divide
  by zero"), `short()` of a vector and `@"name"` (segmentation violations). An
  image called `x.fits` for operand `x` is the image, not the keyword `fits`.
- A string of one letter is a string; IRAF goes looking for an operand.
- `atan2`, `mod` and the 3 to 5 vector `median` take a scalar where a vector could
  go. IRAF reads the scalar's memory as a vector, and corrupts its own or dies.
- Not here: `.` for the last output, which harwood does not keep. And `bwidth`,
  `btype`, `bpixval`, because in IRAF 2.18 they never get to act: a section out of
  the image is refused before they are set, and so it is here.
