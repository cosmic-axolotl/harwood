<!-- from: 06b1950c2d00 -->
# Instalação

O harwood precisa do Python 3.12 ou mais novo. Tudo o que ele usa se instala
com o pip; nada precisa ser configurado depois.

```console
$ pip install "harwood[all]"
```

Os extras, se você preferir escolher:

| extra | o que acrescenta |
|---|---|
| `gui` | a janela: exibição de imagens, o painel Tarefas, o editor |
| `docs` | esta documentação dentro da janela (QtWebEngine); sem ele, as páginas abrem no seu navegador |
| `phot` | scipy, para os ajustes de perfil do imexamine e os filtros de raios cósmicos |
| `ccd` | ccdproc e astroscrappy, os pacotes da comunidade com que a redução de CCD e a limpeza de raios cósmicos do harwood são comparadas; o harwood roda sem eles |
| `fast` | numba, os núcleos compilados |
| `all` | todos os acima, menos `docs` |

## Confira

`harwood doctor` lista o que está instalado, o que falta e o comando que
instala.

```example
title: Está tudo aí?
run: harwood doctor
```

## Abra a janela

```console
$ harwood gui
```

ou `harwood gui obj001.fits` para começar com uma imagem na tela. Sem o extra
`gui`, toda tarefa continua rodando pelo terminal e pelo Python.

## Sem a janela: a sessão

`harwood` sozinho abre uma sessão: o prompt `hw>`, com as tarefas carregadas
uma vez para que cada comando comece na hora. Tab completa, as setas trazem de
volta o que foi digitado (aqui e na janela), e `;` e `&&` põem vários comandos
numa linha. `exit` sai. Veja [a linha de comando](page:app/command-line).

## O idioma

O harwood fala português, inglês e espanhol: a janela, a ajuda, os avisos e as
explicações. Ele começa no idioma do sistema; troque em **Exibir › 🌐 Idioma**,
no topo de cada página da ajuda, ou no terminal:

```console
$ harwood language pt
```

Os nomes das tarefas e dos parâmetros, as descrições dos parâmetros e o que as
tarefas do IRAF imprimem continuam como no IRAF, em inglês, e os arquivos que o
harwood grava também: uma redução se lê do mesmo jeito, seja quem for que a
rodou.
