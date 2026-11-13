<!-- from: 417b2515a01b -->
# La noche de ejemplo

Todos los ejemplos de esta documentación se ejecutan sobre la misma pequeña
noche de datos, que harwood escribe para usted:

```console
$ harwood docs --sample night
$ cd night
```

Se genera en el momento, con los mismos bytes cada vez, así que los números de
estas páginas son los números que usted obtendrá. (La batería de pruebas
ejecuta cada ejemplo sobre ella y lo comprueba.)

| archivo | qué es |
|---|---|
| `bias001.fits` … `bias003.fits` | cuadros de cero: un nivel de 500 ADU y 5 ADU de ruido de lectura |
| `flat001.fits` … `flat003.fits` | flats de cúpula en V, con viñeteo y un patrón píxel a píxel |
| `obj001.fits` … `obj003.fits` | un campo en V de treinta estrellas (FWHM de 3 píxeles) sobre 300 ADU de cielo; el telescopio deriva algunos píxeles entre ellos, y obj003 recibió ocho rayos cósmicos |
| `bias.lis`, `flat.lis`, `obj.lis` | las listas de IRAF de cada uno, para `@bias.lis` |
| `bright.coo` | tres estrellas brillantes de obj001, como `x y` (desde 1) |
| `stars.cur` | un archivo de cursor para imexamine: `r` y `a` sobre las dos estrellas más brillantes |

Cada cuadro tiene 120 × 100 píxeles: la luz cae en `[1:100,1:100]`
(`TRIMSEC`), y las columnas 105 a 120 son una franja de overscan (`BIASSEC`),
como en un CCD de los años 1980. Las cabeceras dicen qué es cada cuadro
(`IMAGETYP`), el filtro (`FILTERS`), la exposición, la ganancia y el ruido de
lectura, la hora y el apuntado.

## Dónde empieza un ejemplo

Cada ejemplo dice de dónde parte: de la noche tal como se escribió, o de la
noche en algún punto de [su reducción](page:start/first-night). Cada estado es
el anterior más estos comandos, ejecutados en la carpeta de la noche (cada
ejemplo también los trae, en *Cómo llegar allí*):

```generated
stages
```

```example
title: La noche de ejemplo, listada
run: harwood ccdlist @bias.lis @flat.lis @obj.lis
output: |
  bias001.fits
  flat001.fits
```
