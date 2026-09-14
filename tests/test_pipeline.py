"""O pré-processamento é aprendido só no treino e lida com nulos."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import construir_preprocessador


def test_imputa_nulos_e_nao_deixa_nan(base_exemplo):
    X = base_exemplo.drop(columns=["alfabetizado"])
    assert X["mun_taxa_ano_anterior"].isna().any(), "a fixture deve conter nulos"
    pre = construir_preprocessador(X)
    Xt = pre.fit_transform(X)
    assert not np.isnan(Xt).any()


def test_indicador_de_faltante_e_criado(base_exemplo):
    X = base_exemplo.drop(columns=["alfabetizado"])
    pre = construir_preprocessador(X)
    pre.fit(X)
    nomes = list(pre.get_feature_names_out())
    assert any("missingindicator" in n.lower() for n in nomes), (
        "a ausência de histórico é informativa e precisa virar variável"
    )


def test_categoria_nunca_vista_no_treino_nao_quebra(base_exemplo):
    X = base_exemplo.drop(columns=["alfabetizado"])
    pre = construir_preprocessador(X)
    pre.fit(X.head(300))
    novo = X.tail(100).copy()
    novo.loc[novo.index[0], "sigla_uf"] = "ZZ"   # UF inexistente
    assert pre.transform(novo).shape[0] == len(novo)


def test_preprocessamento_integrado_ao_modelo(base_exemplo):
    """Requisito do enunciado: pré-processamento dentro do pipeline."""
    X = base_exemplo.drop(columns=["alfabetizado"])
    pipe = Pipeline([
        ("preprocessamento", construir_preprocessador(X)),
        ("modelo", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(X, base_exemplo["alfabetizado"])
    assert pipe.predict_proba(X).shape == (len(X), 2)
