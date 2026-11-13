<!-- from: 417b2515a01b -->
# A noite de exemplo

Todo exemplo desta documentação roda sobre a mesma noitezinha de dados, que o
harwood grava para você:

```console
$ harwood docs --sample night
$ cd night
```

Ela é feita na hora, com os mesmos bytes toda vez, então os números destas
páginas são os números que você vai obter. (A bateria de testes roda cada
exemplo nela e confere.)

| arquivo | o que é |
|---|---|
| `bias001.fits` … `bias003.fits` | quadros de zero: um nível de 500 ADU e 5 ADU de ruído de leitura |
| `flat001.fits` … `flat003.fits` | flats de cúpula em V, com vinhetagem e um padrão pixel a pixel |
| `obj001.fits` … `obj003.fits` | um campo em V de trinta estrelas (FWHM de 3 pixels) sobre 300 ADU de céu; o telescópio deriva alguns pixels entre eles, e o obj003 levou oito raios cósmicos |
| `bias.lis`, `flat.lis`, `obj.lis` | as listas do IRAF de cada um, para `@bias.lis` |
| `bright.coo` | três estrelas brilhantes do obj001, como `x y` (a partir de 1) |
| `stars.cur` | um arquivo de cursor para o imexamine: `r` e `a` nas duas estrelas mais brilhantes |

Todo quadro tem 120 × 100 pixels: a luz cai em `[1:100,1:100]` (`TRIMSEC`), e
as colunas 105 a 120 são uma faixa de overscan (`BIASSEC`), como num CCD dos
anos 1980. Os cabeçalhos dizem o que cada quadro é (`IMAGETYP`), o filtro
(`FILTERS`), a exposição, o ganho e o ruído de leitura, a hora e o
apontamento.

## Onde um exemplo começa

Cada exemplo diz de onde parte: da noite como foi gravada, ou da noite em
algum ponto da [redução dela](page:start/first-night). Cada estado é o
anterior mais estes comandos, rodados na pasta da noite (todo exemplo também
os traz, em *Como chegar lá*):

```generated
stages
```

```example
title: A noite de exemplo, listada
run: harwood ccdlist @bias.lis @flat.lis @obj.lis
output: |
  bias001.fits
  flat001.fits
```
