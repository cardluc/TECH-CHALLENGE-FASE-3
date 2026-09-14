"""Análise exploratória: distribuições, correlações e hipóteses.

Gera os gráficos em images/ e o relatório em reports/eda.md.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.preprocessing.base_analitica import amostrar, construir
# a mesma lista que o modelo usa — duas listas divergiriam com o tempo
from src.preprocessing.pipeline import NUMERICAS_PERMITIDAS as NUMERICAS
from src.utils import get_logger
from src.visualization import graficos

logger = get_logger("eda")


def _tabela(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".2f")


def executar(base: pd.DataFrame | None = None) -> str:
    config.ensure_dirs()
    base = base if base is not None else construir()
    amostra = amostrar(base)

    linhas: list[str] = ["# Análise Exploratória — Alfabetização no 2º ano\n"]
    linhas.append(
        f"Base: **{len(base):,} alunos** avaliados em {config.ANO_ALVO}, "
        f"com contexto educacional de {config.ANO_BASE}. "
        f"Gráficos gerados sobre amostra estratificada de {len(amostra):,} linhas.\n"
        .replace(",", ".")
    )

    # --- alvo ---
    taxa = base["alfabetizado"].mean() * 100
    linhas.append("## 1. O alvo\n")
    linhas.append(
        f"{taxa:.1f}% dos alunos foram classificados como alfabetizados "
        f"(proficiência ≥ 743 na escala Saeb). O alvo é **equilibrado**, o que "
        "dispensa técnicas de balanceamento e permite usar acurácia junto de "
        "métricas mais informativas.\n"
    )
    graficos.distribuicao_alvo(amostra)
    linhas.append("![Distribuição do alvo](../images/01_distribuicao_alvo.png)\n")

    # --- cobertura ---
    cobertura = (base.notna().mean() * 100).round(1).sort_values()
    faltantes = cobertura[cobertura < 100]
    linhas.append("## 2. Valores faltantes\n")
    linhas.append(
        "Os nulos não são ruído: são **ausência de histórico**. Municípios e "
        "escolas que não participaram da avaliação anterior não têm contexto, "
        "e a fonte só publica a distribuição por nível de desempenho a partir "
        f"de {config.ANO_ALVO}.\n"
    )
    linhas.append(_tabela(
        faltantes.reset_index().rename(columns={"index": "variável", 0: "cobertura (%)"})
    ) + "\n")

    # --- categóricas ---
    linhas.append("## 3. Padrões por recorte\n")
    for coluna, titulo, nome in [
        ("regiao", "Alfabetização por região", "05_taxa_regiao"),
        ("sigla_uf", "Alfabetização por UF", "06_taxa_uf"),
        ("rede_descricao", "Alfabetização por rede", "07_taxa_rede"),
    ]:
        graficos.taxa_por_categoria(amostra, coluna, titulo, nome)
        resumo = (
            base.groupby(coluna)["alfabetizado"].agg(["mean", "size"])
            .assign(mean=lambda d: (d["mean"] * 100).round(1))
            .sort_values("mean", ascending=False)
        )
        linhas.append(f"### {titulo}\n")
        linhas.append(f"![{titulo}](../images/{nome}.png)\n")
        if coluna != "sigla_uf":
            linhas.append(_tabela(
                resumo.reset_index().rename(columns={"mean": "% alfabetizados", "size": "alunos"})
            ) + "\n")
        else:
            melhores = resumo.head(3).index.tolist()
            piores = resumo.tail(3).index.tolist()
            linhas.append(
                f"Maiores taxas: **{', '.join(melhores)}**. "
                f"Menores: **{', '.join(piores)}**. "
                f"A diferença entre a melhor e a pior UF é de "
                f"{resumo['mean'].max() - resumo['mean'].min():.1f} pontos percentuais.\n"
            )

    # --- numéricas ---
    linhas.append("## 4. Variáveis numéricas\n")
    graficos.distribuicoes_numericas(amostra, NUMERICAS)
    linhas.append("![Distribuições](../images/02_distribuicoes.png)\n")

    correlacoes = (
        amostra[[c for c in NUMERICAS if c in amostra.columns] + ["alfabetizado"]]
        .corr(numeric_only=True)["alfabetizado"].drop("alfabetizado")
        .sort_values(key=abs, ascending=False).round(3)
    )
    graficos.correlacoes(amostra, NUMERICAS)
    linhas.append("![Correlações](../images/03_correlacoes.png)\n")
    linhas.append("Correlação de cada variável com o alvo:\n")
    linhas.append(_tabela(
        correlacoes.reset_index().rename(columns={"index": "variável", "alfabetizado": "correlação"})
    ) + "\n")

    graficos.relacao_contexto_alvo(amostra)
    linhas.append("## 5. O histórico prevê o futuro?\n")
    linhas.append("![Contexto x alvo](../images/04_contexto_vs_alvo.png)\n")
    linhas.append(
        "A relação é monótona e forte: quanto maior a taxa da escola e do "
        "município no ano anterior, maior a chance de o aluno ser alfabetizado "
        "no ano seguinte. É o que sustenta a viabilidade do modelo.\n"
    )

    linhas.append("## 6. Uma armadilha da fonte: o código da escola\n")
    linhas.append(
        "O microdado traz `id_escola`, mas a coluna é descrita como *máscara "
        "de código fictício* — e a máscara é **regerada a cada ano**. "
        "Evidências: apenas **3,5%** dos códigos presentes nos dois anos "
        "pertencem ao mesmo município, e a taxa de alfabetização por escola "
        "correlaciona apenas **0,25** entre 2023 e 2024, contra **0,64** da "
        "taxa municipal. Usar o histórico da escola seria alimentar o modelo "
        "com ruído. A variável foi descartada; do nível da escola resta "
        "apenas o **porte** medido no próprio ano.\n"
    )

    # --- hipóteses ---
    linhas.append("## 7. Hipóteses analíticas\n")
    linhas.append(
        "1. **O município é a menor unidade de contexto confiável** — o nível "
        "da escola existe no dado, mas não é rastreável entre anos.\n"
        "2. **A desigualdade é regional e estrutural** — a diferença entre UFs "
        "supera a diferença entre redes dentro da mesma UF.\n"
        "3. **A dispersão importa, não só a média** — municípios com desempenho "
        "heterogêneo escondem escolas em risco atrás de uma média aceitável.\n"
        "4. **A meta pactuada carrega informação** — metas mais altas foram "
        "atribuídas a municípios com pior ponto de partida.\n"
        "5. **Sem variáveis individuais, o teto de previsibilidade é o contexto** "
        "— o modelo estima o risco do ambiente em que o aluno está, não o "
        "desempenho pessoal dele.\n"
    )

    destino = config.REPORTS_DIR / "eda.md"
    destino.write_text("\n".join(linhas), encoding="utf-8")
    logger.info("Relatório salvo em %s", destino)
    return str(destino)


if __name__ == "__main__":
    executar()
