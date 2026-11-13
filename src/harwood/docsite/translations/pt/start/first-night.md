<!-- from: 2fc524b6ee8c -->
# A primeira noite

A [noite de exemplo](page:start/sample), dos quadros crus até as magnitudes,
na ordem em que se reduz uma noite de CCD: o bias, o flat, os quadros de
objeto, depois as estrelas. Cada passo é um ou dois comandos; cada exemplo
traz os mesmos passos em Python e em cliques, nas abas. Grave a noite e entre
nela antes:

```console
$ harwood docs --sample night
$ cd night
```

## 1. Veja o que há

`ccdlist` lê os cabeçalhos e diz o que cada quadro é: o tipo pelo `IMAGETYP`,
o filtro pelo `FILTERS`, o tamanho, a exposição.

```example
title: O que a noite tem
run: harwood ccdlist @bias.lis @flat.lis @obj.lis
output: |
  bias001.fits
  obj003.fits
```

## 2. O bias

Um quadro de bias é uma exposição de zero segundos: guarda o deslocamento
eletrônico que o amplificador soma a todo pixel, e nada mais. Cada um primeiro
perde a faixa de overscan, como todo quadro vai perder, e depois os três são
combinados num só, `Zero.fits`, com menos ruído que qualquer um deles.

```example
title: Processar e combinar os quadros de bias
run:
  - harwood ccdproc @bias.lis --overscan --trim --output bias
  - harwood zerocombine bias/bias00*.fits Zero.fits
output: |
  -> Zero.fits
```

`zerocombine` é o [imcombine](task:imcombine) com os ajustes do IRAF para
quadros de bias: ele descarta o pixel mais alto de cada pilha (`minmax`,
`nhigh=1`) antes da média, de modo que um raio cósmico num quadro não chega ao
mestre. `harwood lpar zerocombine` lista esses valores e diz que vieram da
predefinição.

## 3. O flat

Um flat de cúpula é uma exposição iluminada por igual: mostra como cada pixel
responde em comparação com o resto -- a vinhetagem nos cantos, o padrão pixel
a pixel. Os flats recebem o mesmo overscan e corte, e o bias é tirado deles,
antes de serem combinados.

```example
title: Processar e combinar os flats
after: zero
run:
  - harwood ccdproc @flat.lis --overscan --trim --zerocor --zero Zero.fits --output flat
  - harwood flatcombine flat/flat00*.fits Flat.fits
output: |
  V: 3 images -> FlatV.fits
```

```note
Foi pedido `Flat.fits` e foi gravado `FlatV.fits`. O `flatcombine` combina
cada filtro separadamente e põe o filtro no nome, como o IRAF faz, para que
uma noite com flats B e V ganhe `FlatB.fits` e `FlatV.fits` com um só
comando.
```

## 4. Os quadros de objeto

Agora os quadros do céu recebem tudo: overscan, corte, bias, flat. Antes de
rodar, `--plan` diz o que vai fazer com os valores de verdade -- qual faixa de
overscan, qual bias, qual flat -- e não faz nada. Acrescente `--explain` e
cada passo também diz por que está ali.

```example
title: Ver os passos antes
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits --plan
output: |
  overscan: legendre order 1, 1 rejection passes over [105:120,1:100]
  trim: [1:100,1:100]
```

```example
title: Processar os quadros de objeto
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits --output red
output: |
  -> red/obj001.fits
```

Cada quadro processado leva no cabeçalho uma marca de cada passo (`OVERSCAN`,
`TRIM`, `ZEROCOR`, `FLATCOR`), e o harwood não aplica um passo duas vezes: dê
ao ccdproc um quadro de `red/` com as mesmas opções e ele informa quatro
passos pulados, como o IRAF. Deixe de fora o `--output` e os quadros vão para
uma pasta própria sob `output/`; veja
[onde ficam os resultados](page:concepts/outputs).

## 5. Confira o resultado

```example
title: Estatística dos quadros processados
after: reduced
run: harwood imstatistics red/obj00*.fits
output: |
  red/obj001.fits │ 10000 │ 434.777 │ 949.948 │ 239.892 │ 17631.3
```

Dez mil pixels: o quadro de 120 × 100 cortado nos seus 100 × 100 de céu. Para
olhar um quadro, `harwood display red/obj001.fits` o põe na janela, e
`harwood imexamine red/obj001.fits` deixa você apontar para uma estrela e
apertar `r` para ver o perfil e a FWHM, como no IRAF.

## 6. Ache as estrelas

O [daofind](task:daofind) procura picos da largura que você dá. Com a FWHM
definida e o ruído deixado para ele medir, ele acha as estrelas do campo e
nada mais:

```example
title: Achar as estrelas do obj001
after: reduced
run: harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3
output: |
  sources: 24
```

O campo tem trinta estrelas. As que ele não separa estão todas a menos de 4,5
pixels -- uma FWHM e meia -- de uma vizinha, perto o bastante para que duas
estrelas formem um só pico; isso é uma propriedade dos dados, não da busca.

```warning
Dar um `--sigma` menor que o ruído real baixa o limiar junto, e picos de ruído
começam a sair como estrelas: com `--sigma 8` neste quadro, cujo ruído de céu é
de uns 14,5, o daofind informa cinquenta "estrelas", vinte e seis delas ruído.
Deixe-o de fora, como aqui, e o daofind o mede (`sigma:` no que ele imprime).
```

## 7. Meça-as

O [phot](task:phot) põe aberturas em cada estrela, tira o céu de um anel em
volta e grava magnitudes. Aqui, duas aberturas, de 4 e 6 pixels, um anel de
céu de 10 a 15 pixels, o ganho de 2 elétrons por ADU e um ponto zero
arbitrário de 25:

```example
title: Fotometria de abertura do obj001
after: found
run: harwood phot red/obj001.fits obj001.coo obj001.mag --apertures 4,6 --annulus 10 --dannulus 5 --epadu 2 --zmag 25
output: |
  stars: 24
  measured: 24
```

`obj001.mag` é uma tabela que se lê com qualquer ferramenta que leia ECSV, ou
que se abre no painel [Arquivos](page:app/files) da janela: uma linha por
estrela, com o centro, o céu, e o fluxo, a magnitude e o erro em cada
abertura.

## 8. Alinhe e combine

O telescópio derivou entre as três exposições. O [imalign](task:imalign)
centra as três estrelas brilhantes de `bright.coo` em cada quadro, mede o
deslocamento de cada quadro em relação ao obj001 e desloca todos sobre ele:

```example
title: Alinhar os quadros, depois combiná-los
after: reduced
run:
  - harwood imalign red/obj001.fits,red/obj002.fits,red/obj003.fits red/obj001.fits bright.coo --output aligned
  - harwood imcombine aligned/obj00*.fits final.fits
output: |
  red/obj002.fits     -2.297 (0.006)      1.421 (0.006)
  red/obj003.fits      1.100 (0.006)     -3.192 (0.006)
  -> final.fits
```

A noite de exemplo derivou (2,3, −1,4) e (−1,1, 3,2) pixels; o imalign mede os
deslocamentos que desfazem isso, com precisão de milésimos de pixel. Ele
também corta os quadros na parte que os três cobrem, então `final.fits` sai um
pouco menor que os quadros de que veio.

## 9. O que foi feito

Toda execução está no registro, com cada valor e o comando que a repete:

```example
title: O registro da noite
after: reduced
run: harwood log
output: |
  ccdproc
```

`harwood log --replay 3` imprime o comando que repete exatamente a execução 3,
e `harwood log ccdproc --full` lista cada parâmetro de cada execução do
ccdproc e de onde veio o valor dele. A mesma redução agora pode ser escrita
como um [script em Python](page:concepts/python), ou refeita pelo
[painel Tarefas](page:app/clicks) da janela.
