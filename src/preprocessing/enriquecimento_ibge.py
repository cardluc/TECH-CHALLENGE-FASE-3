"""Enriquecimento externo: população e PIB municipal (IBGE via BigQuery).

Rode uma vez; o resultado fica em data/externo_ibge.parquet e é usado
automaticamente pela base analítica.

    "<ID do seu projeto>" | Set-Content .gcp_project -NoNewline
    python -m src.preprocessing.enriquecimento_ibge

O ID também pode vir de GCP_PROJECT_ID, que tem precedência sobre o arquivo.
"""
from __future__ import annotations

from src import config
from src.utils import get_logger

logger = get_logger("enriquecimento")

# Cada fonte tem seu próprio teto, e o critério é a DATA DE PUBLICAÇÃO, não o
# ano de referência. A estimativa populacional é publicada no próprio ano; o PIB
# dos Municípios sai com ~2 anos de atraso, então o PIB de referência ANO_BASE
# ainda não existia quando a previsão seria feita. Sem essa distinção, o projeto
# usaria um número que só passou a existir depois da prova que ele prevê.
# Os anos efetivos ficam gravados no arquivo e são reconferidos na leitura.
CONSULTA = """
WITH pop AS (
    SELECT id_municipio, ano, populacao
    FROM `basedosdados.br_ibge_populacao.municipio`
    WHERE ano = (SELECT MAX(ano) FROM `basedosdados.br_ibge_populacao.municipio`
                 WHERE ano <= {ano_pop})
),
pib AS (
    SELECT id_municipio, ano, pib
    FROM `basedosdados.br_ibge_pib.municipio`
    WHERE ano = (SELECT MAX(ano) FROM `basedosdados.br_ibge_pib.municipio`
                 WHERE ano <= {ano_pib})
)
SELECT
    pop.id_municipio,
    pop.ano AS ano_populacao,
    pib.ano AS ano_pib,
    pop.populacao,
    pib.pib,
    SAFE_DIVIDE(pib.pib, pop.populacao) AS pib_per_capita
FROM pop
LEFT JOIN pib USING (id_municipio)
"""


def main() -> None:
    if not config.BQ_BILLING_PROJECT:
        raise RuntimeError("Defina GCP_PROJECT_ID com o ID do seu projeto do BigQuery.")
    import pandas_gbq

    config.ensure_dirs()
    teto_pib = config.ANO_BASE - config.DEFASAGEM_PUBLICACAO_PIB
    df = pandas_gbq.read_gbq(
        CONSULTA.format(ano_pop=config.ANO_BASE, ano_pib=teto_pib),
        project_id=config.BQ_BILLING_PROJECT, progress_bar_type=None,
    )
    # a fonte traz um registro residual sem código de município
    df = df.dropna(subset=["id_municipio"])
    df["id_municipio"] = df["id_municipio"].astype("string").str.zfill(7)
    df.to_parquet(config.ARQUIVO_IBGE, index=False)
    logger.info(
        "%d municípios | população de %s (teto %d), PIB de %s (teto %d, defasagem "
        "de publicação de %d anos)",
        len(df), df["ano_populacao"].max(), config.ANO_BASE,
        df["ano_pib"].max(), teto_pib, config.DEFASAGEM_PUBLICACAO_PIB,
    )


if __name__ == "__main__":
    main()
