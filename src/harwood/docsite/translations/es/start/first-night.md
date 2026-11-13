<!-- from: 2fc524b6ee8c -->
# La primera noche

La [noche de ejemplo](page:start/sample), desde los cuadros crudos hasta las
magnitudes, en el orden en que se reduce una noche de CCD: el bias, el flat,
los cuadros de objeto, después las estrellas. Cada paso es uno o dos comandos;
cada ejemplo trae los mismos pasos en Python y en clics, en sus pestañas.
Escriba la noche y entre en ella primero:

```console
$ harwood docs --sample night
$ cd night
```

## 1. Vea lo que hay

`ccdlist` lee las cabeceras y dice qué es cada cuadro: el tipo por
`IMAGETYP`, el filtro por `FILTERS`, el tamaño, la exposición.

```example
title: Qué tiene la noche
run: harwood ccdlist @bias.lis @flat.lis @obj.lis
output: |
  bias001.fits
  obj003.fits
```

## 2. El bias

Un cuadro de bias es una exposición de cero segundos: guarda el desplazamiento
electrónico que el amplificador suma a cada píxel, y nada más. A cada uno
primero se le quita la franja de overscan, como a todo cuadro, y después los
tres se combinan en uno, `Zero.fits`, con menos ruido que cualquiera de ellos.

```example
title: Procesar y combinar los cuadros de bias
run:
  - harwood ccdproc @bias.lis --overscan --trim --output bias
  - harwood zerocombine bias/bias00*.fits Zero.fits
output: |
  -> Zero.fits
```

`zerocombine` es [imcombine](task:imcombine) con los ajustes de IRAF para
cuadros de bias: descarta el píxel más alto de cada pila (`minmax`,
`nhigh=1`) antes de promediar, de modo que un rayo cósmico en un cuadro no
llega al maestro. `harwood lpar zerocombine` lista esos valores y dice que
vinieron del preajuste.

## 3. El flat

Un flat de cúpula es una exposición iluminada de manera uniforme: muestra cómo
responde cada píxel comparado con el resto -- el viñeteo en las esquinas, el
patrón píxel a píxel. Los flats reciben el mismo overscan y recorte, y se les
quita el bias, antes de combinarlos.

```example
title: Procesar y combinar los flats
after: zero
run:
  - harwood ccdproc @flat.lis --overscan --trim --zerocor --zero Zero.fits --output flat
  - harwood flatcombine flat/flat00*.fits Flat.fits
output: |
  V: 3 images -> FlatV.fits
```

```note
Se pidió `Flat.fits` y se escribió `FlatV.fits`. `flatcombine` combina cada
filtro por separado y pone el filtro en el nombre, como IRAF, para que una
noche con flats B y V obtenga `FlatB.fits` y `FlatV.fits` con un solo comando.
```

## 4. Los cuadros de objeto

Ahora los cuadros del cielo reciben todo: overscan, recorte, bias, flat. Antes
de ejecutarlo, `--plan` dice lo que va a hacer con los valores reales -- qué
franja de overscan, qué bias, qué flat -- y no hace nada. Agregue `--explain`
y cada paso dice también por qué está ahí.

```example
title: Ver los pasos primero
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits --plan
output: |
  overscan: legendre order 1, 1 rejection passes over [105:120,1:100]
  trim: [1:100,1:100]
```

```example
title: Procesar los cuadros de objeto
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits --output red
output: |
  -> red/obj001.fits
```

Cada cuadro procesado lleva en su cabecera una marca de cada paso
(`OVERSCAN`, `TRIM`, `ZEROCOR`, `FLATCOR`), y harwood no aplica un paso dos
veces: dele a ccdproc un cuadro de `red/` con las mismas opciones y informa
cuatro pasos omitidos, como IRAF. Si deja fuera `--output`, los cuadros van a
una carpeta propia bajo `output/`; vea
[dónde quedan los resultados](page:concepts/outputs).

## 5. Compruebe el resultado

```example
title: Estadística de los cuadros procesados
after: reduced
run: harwood imstatistics red/obj00*.fits
output: |
  red/obj001.fits │ 10000 │ 434.777 │ 949.948 │ 239.892 │ 17631.3
```

Diez mil píxeles: el cuadro de 120 × 100 recortado a sus 100 × 100 de cielo.
Para mirar un cuadro, `harwood display red/obj001.fits` lo pone en la ventana,
y `harwood imexamine red/obj001.fits` le permite apuntar a una estrella y
pulsar `r` para ver su perfil y su FWHM, como en IRAF.

## 6. Encuentre las estrellas

[daofind](task:daofind) busca picos del ancho que usted le da. Con la FWHM
fijada y el ruido dejado para que lo mida, encuentra las estrellas del campo y
nada más:

```example
title: Encontrar las estrellas de obj001
after: reduced
run: harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3
output: |
  sources: 24
```

El campo tiene treinta estrellas. Las que no separa están todas a menos de 4,5
píxeles -- una FWHM y media -- de una vecina, lo bastante cerca como para que
dos estrellas formen un solo pico; eso es una propiedad de los datos, no de la
búsqueda.

```warning
Dar un `--sigma` menor que el ruido real baja el umbral con él, y los picos de
ruido empiezan a salir como estrellas: con `--sigma 8` en este cuadro, cuyo
ruido de cielo es de unos 14,5, daofind informa cincuenta "estrellas",
veintiséis de ellas ruido. Déjelo fuera, como aquí, y daofind lo mide
(`sigma:` en lo que imprime).
```

## 7. Mídalas

[phot](task:phot) pone aperturas en cada estrella, toma el cielo de un anillo
alrededor y escribe magnitudes. Aquí, dos aperturas, de 4 y 6 píxeles, un
anillo de cielo de 10 a 15 píxeles, la ganancia de 2 electrones por ADU y un
punto cero arbitrario de 25:

```example
title: Fotometría de apertura de obj001
after: found
run: harwood phot red/obj001.fits obj001.coo obj001.mag --apertures 4,6 --annulus 10 --dannulus 5 --epadu 2 --zmag 25
output: |
  stars: 24
  measured: 24
```

`obj001.mag` es una tabla que se lee con cualquier herramienta que lea ECSV, o
que se abre en el panel [Archivos](page:app/files) de la ventana: una fila por
estrella, con el centro, el cielo, y el flujo, la magnitud y el error en cada
apertura.

## 8. Alinee y combine

El telescopio derivó entre las tres exposiciones. [imalign](task:imalign)
centra las tres estrellas brillantes de `bright.coo` en cada cuadro, mide el
desplazamiento de cada cuadro respecto de obj001 y los desplaza a todos sobre
él:

```example
title: Alinear los cuadros, después combinarlos
after: reduced
run:
  - harwood imalign red/obj001.fits,red/obj002.fits,red/obj003.fits red/obj001.fits bright.coo --output aligned
  - harwood imcombine aligned/obj00*.fits final.fits
output: |
  red/obj002.fits     -2.297 (0.006)      1.421 (0.006)
  red/obj003.fits      1.100 (0.006)     -3.192 (0.006)
  -> final.fits
```

La noche de ejemplo derivó (2,3, −1,4) y (−1,1, 3,2) píxeles; imalign mide los
desplazamientos que lo deshacen, con precisión de milésimas de píxel. También
recorta los cuadros a la parte que cubren los tres, así que `final.fits` sale
un poco más chico que los cuadros de los que vino.

## 9. Lo que se hizo

Toda ejecución está en el registro, con cada valor y el comando que la repite:

```example
title: El registro de la noche
after: reduced
run: harwood log
output: |
  ccdproc
```

`harwood log --replay 3` imprime el comando que repite exactamente la
ejecución 3, y `harwood log ccdproc --full` lista cada parámetro de cada
ejecución de ccdproc y de dónde vino su valor. La misma reducción ahora se
puede escribir como un [script de Python](page:concepts/python), o rehacer
desde el [panel Tareas](page:app/clicks) de la ventana.
