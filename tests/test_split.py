"""A separação treino/teste é feita por município."""
from __future__ import annotations

from src.modeling.treinar import GRUPO, separar_treino_teste


def test_nenhum_municipio_nos_dois_conjuntos(base_exemplo):
    treino, teste = separar_treino_teste(base_exemplo)
    assert not set(treino[GRUPO]) & set(teste[GRUPO])


def test_split_cobre_toda_a_base(base_exemplo):
    treino, teste = separar_treino_teste(base_exemplo)
    assert len(treino) + len(teste) == len(base_exemplo)
    assert len(teste) > 0 and len(treino) > 0
