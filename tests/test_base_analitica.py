"""Regras da construção da base."""
from __future__ import annotations

from src.preprocessing.base_analitica import REGIAO, amostrar


def test_todas_as_ufs_tem_regiao():
    assert len(REGIAO) == 27


def test_amostra_preserva_proporcao_do_alvo(base_exemplo):
    amostra = amostrar(base_exemplo, n=200, seed=0)
    original = base_exemplo["alfabetizado"].mean()
    assert abs(amostra["alfabetizado"].mean() - original) < 0.05
    assert len(amostra) <= 220


def test_amostra_menor_que_pedido_retorna_tudo(base_exemplo):
    assert len(amostrar(base_exemplo, n=10_000)) == len(base_exemplo)
