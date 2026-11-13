<!-- from: 06b1950c2d00 -->
# Instalación

harwood necesita Python 3.12 o más nuevo. Todo lo que usa se instala con pip;
no hay nada que configurar después.

```console
$ pip install "harwood[all]"
```

Los extras, si prefiere elegir:

| extra | qué agrega |
|---|---|
| `gui` | la ventana: visualización de imágenes, el panel Tareas, el editor |
| `docs` | esta documentación dentro de la ventana (QtWebEngine); sin él, las páginas se abren en su navegador |
| `phot` | scipy, para los ajustes de perfil de imexamine y los filtros de rayos cósmicos |
| `ccd` | ccdproc y astroscrappy, los paquetes de la comunidad con los que se comparan la reducción de CCD y la limpieza de rayos cósmicos de harwood; harwood funciona sin ellos |
| `fast` | numba, los núcleos compilados |
| `all` | todos los anteriores, menos `docs` |

## Compruébelo

`harwood doctor` lista lo que está instalado, lo que falta y el comando que lo
instala.

```example
title: ¿Está todo?
run: harwood doctor
```

## Abra la ventana

```console
$ harwood gui
```

o `harwood gui obj001.fits` para empezar con una imagen en pantalla. Sin el
extra `gui`, toda tarea sigue funcionando desde la terminal y desde Python.

## Sin la ventana: la sesión

`harwood` solo abre una sesión: el prompt `hw>`, con las tareas cargadas una
vez para que cada comando empiece enseguida. Tab completa, las flechas
recuperan lo escrito (aquí y en la ventana), y `;` y `&&` ponen varios
comandos en una línea. `exit` sale. Vea [la línea de comandos](page:app/command-line).

## El idioma

harwood habla portugués, inglés y español: la ventana, la ayuda, los avisos y
las explicaciones. Empieza en el idioma del sistema; cámbielo en
**Ver › 🌐 Idioma**, arriba de cada página de la ayuda, o en la terminal:

```console
$ harwood language es
```

Los nombres de las tareas y de los parámetros, las descripciones de los
parámetros y lo que imprimen las tareas de IRAF siguen como en IRAF, en
inglés, y también los archivos que escribe harwood: una reducción se lee igual,
sea quien sea que la haya ejecutado.
