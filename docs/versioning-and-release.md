# Docs Versioning and Release

As docs são publicadas por versão usando tags Git e `mike`.

## Como publicar uma nova versão

1. Atualize código e docs no branch principal.
2. Atualize a versão em todos os arquivos gerenciados:

```bash
python scripts/update_version.py 0.3.4
```

3. Crie tag no formato `vX.Y.Z` (semver).
4. Faça push da tag.

```bash
git tag v0.3.4
git push origin v0.3.4
```

## O que o pipeline faz

- workflow acionado apenas por `push` de tags de versão (`vX.Y.Z`, com opcional sufixo);
- build `mkdocs --strict`;
- deploy em `gh-pages` com:
  - versão numérica (`0.3.0`);
  - alias `latest`;
- mantém versões antigas no branch `gh-pages`.

> Não há deploy de docs em push de branch/pull request.
> Sem tag de release, o site publicado não muda.

## Seleção de versão na UI

A UI usa `Material for MkDocs` + `mike` com seletor de versão habilitado.

Você consegue alternar entre:

- `latest` (versão padrão);
- releases antigas (`0.2.0`, `0.2.1`, etc.).

## Rollback simples

Se uma versão publicada ficou ruim:

1. ajuste docs no commit correto;
2. publique novo patch (`vX.Y.(Z+1)`).

Evite reusar tag já publicada para não quebrar histórico.
