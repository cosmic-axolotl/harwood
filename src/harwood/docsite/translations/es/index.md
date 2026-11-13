<!-- from: 70eec3bf1809 -->
# harwood

Las tareas que conoce de IRAF -- `ccdproc`, `imcombine`, `daofind`, `phot`,
`apall`, `identify` -- con los mismos nombres y los mismos parámetros, en
Python, con una ventana para mirar los datos y nada que configurar.

Toda tarea se puede hacer de tres maneras, y son lo mismo:

- **escrita**, en una terminal o en el prompt de la ventana:
  `harwood daofind obj001.fits --fwhmpsf 3`;
- **desde Python**: `harwood.run("daofind", image="obj001.fits", fwhmpsf=3.0)`;
- **con clics**, en el panel Tareas de la ventana, donde el formulario muestra
  el comando al que equivale mientras lo completa.

Los ejemplos de esta documentación muestran las tres. Elija la pestaña que
lea mejor; la elección lo acompaña de página en página.

```example
title: Qué hay en una noche
run: harwood ccdlist @obj.lis
output: |
  obj001.fits
```

## Por dónde empezar

- Nuevo en harwood: [instálelo](page:start/install), obtenga
  [la noche de ejemplo](page:start/sample) y [redúzcala](page:start/first-night)
  desde los cuadros crudos hasta las magnitudes. Lleva unos pocos minutos.
- Viniendo de IRAF: [qué cambió y qué no](page:iraf/migration). La versión
  corta: los nombres son los mismos, `epar` es `--editpar`, `lpar` es `lpar`,
  y no hay paquetes que cargar.
- Buscando una tarea: la columna de la izquierda es el árbol de paquetes de
  IRAF, `noao › digiphot › apphot › daofind`, o busque con `/`.

## Los cuidados de harwood

- **Dice lo que va a hacer antes de hacerlo.** `--plan` lista los pasos con
  los valores reales y se detiene; `--explain` dice por qué está cada paso.
  Vea [los comandos](page:concepts/commands).
- **Anota lo que hizo.** Cada ejecución tiene su propia carpeta con un
  `run.yaml`, una entrada en el registro y HISTORY en las cabeceras que
  escribió. Vea [dónde quedan los resultados](page:concepts/outputs).
- **Todo aviso dice qué hacer.** Y nunca solo con el color. Vea
  [avisos](page:concepts/notices).
- **Se compara con IRAF.** Las tareas portadas se comparan con la salida del
  propio IRAF 2.18, número por número, en la batería de pruebas.
