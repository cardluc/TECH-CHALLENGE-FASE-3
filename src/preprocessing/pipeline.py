"""Pré-processamento integrado ao modelo (ColumnTransformer).

Tudo que aprende parâmetro a partir dos dados — mediana da imputação,
média e desvio da padronização, categorias do encoding — fica dentro do
Pipeline. Assim esses parâmetros são estimados apenas no conjunto de
treino, em cada dobra da validação cruzada, sem vazar o teste.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.utils import get_logger

logger = get_logger("pipeline")

# Lista BRANCA: só entra no modelo o que está aqui. Uma lista negra deixaria
# passar qualquer coluna nova, inclusive uma que vaze o resultado do ano-alvo.
NUMERICAS_PERMITIDAS = [
    "mun_taxa_ano_anterior", "mun_media_ano_anterior", "mun_desvio_ano_anterior",
    "mun_p25_ano_anterior", "mun_p75_ano_anterior", "mun_alunos_ano_anterior",
    "uf_taxa_ano_anterior", "uf_media_ano_anterior",
    "meta_do_ano", "meta_uf_do_ano", "dist_meta_partida",
    "dist_meta_mun_uf", "dist_meta_mun_nacional",
    "populacao", "pib", "pib_per_capita",
]

CATEGORICAS = ["rede_descricao", "sigla_uf", "regiao", "caderno"]

# Identificadores, alvo e peso amostral nunca são atributos preditivos.
COLUNAS_FORA = {"id_aluno", "id_escola", "id_municipio", "alfabetizado", "peso_aluno"}

# populacao e pib_per_capita são fortemente assimétricos; o escalonamento
# robusto (mediana e IQR) é menos sensível a esses extremos que o z-score
ESCALADORES = {"padrao": StandardScaler, "robusto": RobustScaler}


def separar_colunas(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categoricas = [c for c in CATEGORICAS if c in df.columns]
    numericas = [
        c for c in NUMERICAS_PERMITIDAS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    ignoradas = set(df.columns) - set(numericas) - set(categoricas) - COLUNAS_FORA
    if ignoradas:
        logger.warning("Colunas fora da lista branca, descartadas: %s", sorted(ignoradas))
    return numericas, categoricas


def construir_preprocessador(df: pd.DataFrame, escalador: str = "padrao") -> ColumnTransformer:
    numericas, categoricas = separar_colunas(df)

    # add_indicator: a ausência de histórico é informativa, não ruído
    numerico = Pipeline([
        ("imputacao", SimpleImputer(strategy="median", add_indicator=True)),
        ("escala", ESCALADORES[escalador]()),
    ])
    categorico = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                   sparse_output=False)),
    ])

    return ColumnTransformer(
        [("num", numerico, numericas), ("cat", categorico, categoricas)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def nomes_features(preprocessador: ColumnTransformer) -> list[str]:
    return list(preprocessador.get_feature_names_out())
