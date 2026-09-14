"""Pipeline reproduzível de ponta a ponta.

    python main.py                 # tudo
    python main.py --etapa eda     # apenas uma etapa

Etapas: base -> eda -> treino -> avaliacao -> estrategia -> relatorio
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger("main")
ETAPAS = ["base", "eda", "treino", "avaliacao", "estrategia", "relatorio"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de ML — alfabetização")
    parser.add_argument("--etapa", choices=ETAPAS, help="executa apenas uma etapa")
    args = parser.parse_args()
    etapas = [args.etapa] if args.etapa else ETAPAS
    config.ensure_dirs()

    from src.preprocessing.base_analitica import amostrar, construir

    base = None
    if {"base", "eda", "treino"} & set(etapas):
        logger.info("==== base analítica ====")
        base = construir()
        base.to_parquet(config.DATA_DIR / "base_analitica.parquet", index=False)

    if "eda" in etapas:
        logger.info("==== análise exploratória ====")
        from src.preprocessing.eda import executar as rodar_eda
        rodar_eda(base)

    if "treino" in etapas:
        logger.info("==== treinamento ====")
        from src.modeling.treinar import treinar
        treinar(amostrar(base))

    if "avaliacao" in etapas:
        logger.info("==== avaliação ====")
        from src.evaluation.avaliar import executar as avaliar
        avaliar()

    if "estrategia" in etapas:
        logger.info("==== aplicação estratégica ====")
        from src.evaluation.aplicacao_estrategica import executar as estrategia
        modelo = joblib.load(config.MODELS_DIR / "modelo.joblib")
        amostra = amostrar(construir() if base is None else base, n=600_000)
        dim = pd.read_parquet(config.FASE2_DATA / "silver/dim_municipio")
        estrategia(modelo, amostra, dim)

    if "relatorio" in etapas:
        logger.info("==== relatório ====")
        from src.relatorio import gerar
        gerar()

    logger.info("Concluído.")


if __name__ == "__main__":
    main()
