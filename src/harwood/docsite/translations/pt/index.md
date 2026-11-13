<!-- from: 70eec3bf1809 -->
# harwood

As tarefas que você conhece do IRAF -- `ccdproc`, `imcombine`, `daofind`,
`phot`, `apall`, `identify` -- com os mesmos nomes e os mesmos parâmetros, em
Python, com uma janela para olhar os dados e nada para configurar.

Toda tarefa pode ser feita de três jeitos, e eles são a mesma coisa:

- **digitada**, num terminal ou no prompt da janela:
  `harwood daofind obj001.fits --fwhmpsf 3`;
- **pelo Python**: `harwood.run("daofind", image="obj001.fits", fwhmpsf=3.0)`;
- **com cliques**, no painel Tarefas da janela, onde o formulário mostra o
  comando a que ele equivale enquanto você o preenche.

Os exemplos desta documentação mostram os três. Escolha a aba que você lê
melhor; a escolha acompanha você de página em página.

```example
title: O que há numa noite
run: harwood ccdlist @obj.lis
output: |
  obj001.fits
```

## Por onde começar

- Novo no harwood: [instale](page:start/install), pegue
  [a noite de exemplo](page:start/sample) e [reduza-a](page:start/first-night)
  dos quadros crus até as magnitudes. Leva poucos minutos.
- Vindo do IRAF: [o que mudou e o que não mudou](page:iraf/migration). A
  versão curta: os nomes são os mesmos, `epar` é `--editpar`, `lpar` é
  `lpar`, e não há pacotes para carregar.
- Procurando uma tarefa: a coluna à esquerda é a árvore de pacotes do IRAF,
  `noao › digiphot › apphot › daofind`, ou busque com `/`.

## Os cuidados do harwood

- **Ele diz o que vai fazer antes de fazer.** `--plan` lista os passos com os
  valores de verdade e para; `--explain` diz por que cada passo está ali.
  Veja [os comandos](page:concepts/commands).
- **Ele anota o que fez.** Cada execução ganha uma pasta própria com um
  `run.yaml`, uma entrada no registro e HISTORY nos cabeçalhos que gravou.
  Veja [onde ficam os resultados](page:concepts/outputs).
- **Todo aviso diz o que fazer.** E nunca só pela cor. Veja
  [avisos](page:concepts/notices).
- **Ele é conferido contra o IRAF.** As tarefas portadas são comparadas com a
  saída do próprio IRAF 2.18, número a número, na bateria de testes.
