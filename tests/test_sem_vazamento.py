"""O alvo não pode ser reconstruível a partir das features."""
from __future__ import annotations

from src.preprocessing.pipeline import COLUNAS_FORA, separar_colunas

# Colunas que revelam o resultado do próprio ano — nenhuma pode virar feature.
PROIBIDAS = {
    "proficiencia",           # o alvo é proficiencia >= 743
    "alfabetizado_descricao",
    "indicador_alfabetizacao",
    "taxa_oficial",
    "mun_taxa_ano_atual",
    # a Gold já traz o resultado calculado na partição do ano-alvo
    "indicador_alfabetizacao",
    "gap_meta",
    "atingiu_meta",
    "target_atingiu_meta",
    "divergencia_oficial",
}


def test_nenhuma_coluna_do_ano_alvo_vira_feature(base_exemplo):
    numericas, categoricas = separar_colunas(base_exemplo)
    assert PROIBIDAS.isdisjoint(set(numericas) | set(categoricas))


def test_coluna_proibida_injetada_e_rejeitada(base_exemplo):
    """Contraprova: a lista branca precisa barrar colunas que não conhece."""
    df = base_exemplo.copy()
    for proibida in PROIBIDAS:
        df[proibida] = 1.0
    numericas, categoricas = separar_colunas(df)
    assert PROIBIDAS.isdisjoint(set(numericas) | set(categoricas)), (
        "coluna proibida passou: a seleção precisa ser lista branca, não lista negra"
    )


def test_identificadores_e_peso_fora_das_features(base_exemplo):
    numericas, categoricas = separar_colunas(base_exemplo)
    usadas = set(numericas) | set(categoricas)
    assert usadas.isdisjoint(COLUNAS_FORA)
    assert "peso_aluno" not in usadas, "o peso amostral é ponderação, não atributo"


def test_features_sao_todas_do_ano_anterior_ou_pactuadas(base_exemplo):
    numericas, _ = separar_colunas(base_exemplo)
    for coluna in numericas:
        assert (
            coluna.endswith("_ano_anterior")
            or coluna in {"meta_do_ano", "meta_uf_do_ano", "dist_meta_partida",
                          "dist_meta_mun_uf", "dist_meta_mun_nacional",
                          "populacao", "pib", "pib_per_capita"}
        ), f"{coluna} não é claramente anterior ao resultado"
