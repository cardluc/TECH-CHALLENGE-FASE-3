"""Monta a base analítica por aluno a partir da camada Gold da Fase 2.

Unidade de análise: um aluno avaliado no ano-alvo.
Alvo: alfabetizado (1/0), conforme o ponto de corte de 743 pontos do Saeb.

Origem dos dados — camada Gold (`gold_*`), exceto três casos justificados:
  · o microdado por aluno, que a Gold agrega e portanto não serve de alvo;
  · a dispersão do desempenho municipal (desvio, p25, p75), único agregado
    que a Gold não materializa;
  · a taxa oficial na granularidade completa, usada como reserva do ICA
    (ver `contexto_municipio`).

Regra central contra data leakage: NENHUMA feature pode conter informação
do resultado do próprio ano. A Gold exige atenção redobrada aqui, porque ela
JÁ TRAZ o resultado calculado — `indicador_alfabetizacao`, `taxa_oficial`,
`gap_meta` e `atingiu_meta` existem na partição do ano-alvo. Por isso toda
leitura de contexto é filtrada em ANO_BASE, e das partições do ano-alvo só se
extrai a meta, que é pactuada com antecedência.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger("base_analitica")

REGIAO = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte",
    "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

# A fonte só publica a distribuição por nível de desempenho a partir de
# 2024 — em 2023 essas colunas são nulas. A dispersão do desempenho é
# reconstruída do microdado do ano anterior (ver contexto_municipio).


REDE_PUBLICA = "Pública"


def _ler(caminho: str, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(config.FASE2_DATA / caminho, **kwargs)


def _gold(nome: str) -> pd.DataFrame:
    """Tabela Gold da Fase 2. A partição `ano` volta como texto."""
    df = _ler(f"gold/{nome}")
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"].astype("string"), errors="coerce").astype("int64")
    return df


def _alunos(ano: int, colunas: list[str]) -> pd.DataFrame:
    df = _ler(f"silver/fato_aluno/ano={ano}", columns=colunas)
    return df[df["registro_valido"] & (df["serie"] == config.SERIE_ALVO)]


# Por que não existe nenhuma feature de escola aqui:
#
# 1. Histórico da escola é impossível — o `id_escola` é máscara regerada a cada
#    ano. Só 3,5% dos códigos repetidos entre 2023 e 2024 pertencem ao mesmo
#    município, e a taxa da escola correlaciona 0,25 entre anos, contra 0,64
#    da taxa municipal.
# 2. Porte da escola no ano-alvo também não serve: contar alunos avaliados em
#    2024 só é possível depois da prova. Não é vazamento (não contém o alvo),
#    é indisponibilidade na data em que a previsão seria feita.
#
# Uma feature de escola exigiria o Censo Escolar, que traz infraestrutura e
# corpo docente conhecidos antes do ciclo. Ver "evoluções futuras" no README.


def contexto_municipio() -> pd.DataFrame:
    """Indicador Criança Alfabetizada do município no ano anterior + dispersão.

    O ICA vem da Gold, onde é definido uma única vez (média ponderada por
    `peso_aluno`). A Gold é construída A PARTIR do microdado, então um
    município sem aluno avaliado no ano-base simplesmente não tem linha lá —
    são 23% dos registros. Para esses entra a taxa oficial publicada pelo
    INEP, a mesma que alimenta a coluna `taxa_oficial` da Gold, lida na
    granularidade completa. Sem essa reserva a variável perderia 23% de
    cobertura e a validação do produto, 521 unidades.

    A dispersão é reconstruída do microdado porque a Gold agrega e não guarda
    quartis.
    """
    gold = _gold("gold_indicador_municipio")
    ica = gold[gold["ano"] == config.ANO_BASE][
        ["id_municipio", "rede_descricao", "indicador_alfabetizacao", "proficiencia_media"]
    ].rename(columns={"proficiencia_media": "mun_media_ano_anterior"})

    oficial = _ler("silver/oficial_municipio")
    oficial = oficial[
        (oficial["ano"] == config.ANO_BASE) & (~oficial["rede_agregada"])
    ][["id_municipio", "rede_descricao", "taxa_alfabetizacao"]]

    ica = ica.merge(oficial, on=["id_municipio", "rede_descricao"], how="outer")
    ica["mun_taxa_ano_anterior"] = ica["indicador_alfabetizacao"].fillna(
        ica["taxa_alfabetizacao"])
    ica = ica.drop(columns=["indicador_alfabetizacao", "taxa_alfabetizacao"])

    micro = _alunos(config.ANO_BASE, [
        "id_municipio", "rede_descricao", "proficiencia", "registro_valido", "serie",
    ])
    disp = micro.groupby(["id_municipio", "rede_descricao"])["proficiencia"].agg(
        mun_desvio_ano_anterior="std",
        mun_p25_ano_anterior=lambda s: s.quantile(0.25),
        mun_p75_ano_anterior=lambda s: s.quantile(0.75),
        mun_alunos_ano_anterior="size",
    ).round(2).reset_index()

    return ica.merge(disp, on=["id_municipio", "rede_descricao"], how="outer")


def contexto_uf() -> pd.DataFrame:
    """ICA da UF no ano anterior, por rede."""
    gold = _gold("gold_indicador_uf")
    return gold[gold["ano"] == config.ANO_BASE][
        ["sigla_uf", "rede_descricao", "indicador_alfabetizacao", "proficiencia_media"]
    ].rename(columns={
        "indicador_alfabetizacao": "uf_taxa_ano_anterior",
        "proficiencia_media": "uf_media_ano_anterior",
    })


def metas_municipais() -> pd.DataFrame:
    """Meta municipal pactuada para o ano-alvo.

    Da partição do ano-alvo só sai a meta: `indicador_alfabetizacao`,
    `gap_meta` e `atingiu_meta` da mesma linha são o resultado e ficam fora.
    """
    gold = _gold("gold_indicador_municipio")
    return gold[gold["ano"] == config.ANO_ALVO][
        ["id_municipio", "rede_descricao", "meta"]
    ].rename(columns={"meta": "meta_do_ano"})


def metas_estaduais() -> pd.DataFrame:
    """Meta estadual pactuada para o ano-alvo, por rede.

    As metas de UF são pactuadas para a rede Pública consolidada; a linha da
    rede específica do aluno vem primeiro e a Pública entra como reserva.
    """
    gold = _gold("gold_indicador_uf")
    alvo = gold[gold["ano"] == config.ANO_ALVO][["sigla_uf", "rede_descricao", "meta"]]
    publica = alvo[alvo["rede_descricao"] == REDE_PUBLICA].set_index("sigla_uf")["meta"]
    por_rede = alvo[alvo["rede_descricao"] != REDE_PUBLICA].copy()
    por_rede["meta"] = por_rede["meta"].fillna(por_rede["sigla_uf"].map(publica))
    return por_rede.rename(columns={"meta": "meta_uf_do_ano"})


def meta_nacional() -> float | None:
    """Meta nacional da rede Pública para o ano-alvo.

    É um escalar: como atributo por aluno não teria variância nenhuma. Entra
    apenas como régua, para medir o quanto a meta local se afasta dela.
    """
    gold = _gold("gold_evolucao_brasil")
    linha = gold[(gold["ano"] == config.ANO_ALVO)
                 & (gold["rede_descricao"] == REDE_PUBLICA)]["meta"].dropna()
    return float(linha.iloc[0]) if len(linha) else None


PROCEDENCIA_IBGE = ["ano_populacao", "ano_pib"]


def enriquecimento_externo() -> pd.DataFrame | None:
    """População e PIB municipal (IBGE), se já tiverem sido baixados.

    Um arquivo sem `ano_populacao` e `ano_pib` foi gerado por uma versão da
    consulta que não fixava o ano. Sem esses campos não dá para verificar se o
    dado é anterior à data em que a previsão seria feita — e um enriquecimento
    que não se pode auditar não entra calado. O arquivo é recusado.
    """
    if not config.ARQUIVO_IBGE.exists():
        logger.warning(
            "Enriquecimento IBGE ausente (%s) — rode src/preprocessing/enriquecimento_ibge.py",
            config.ARQUIVO_IBGE.name,
        )
        return None

    df = pd.read_parquet(config.ARQUIVO_IBGE)
    faltando = [c for c in PROCEDENCIA_IBGE if c not in df.columns]
    if faltando:
        logger.error(
            "Enriquecimento IBGE recusado: %s sem %s. Regere com "
            "src/preprocessing/enriquecimento_ibge.py.",
            config.ARQUIVO_IBGE.name, faltando,
        )
        return None

    # cada fonte tem seu teto: a população sai no próprio ano de referência,
    # o PIB municipal só é publicado anos depois (ver config)
    tetos = {
        "ano_populacao": config.ANO_BASE,
        "ano_pib": config.ANO_BASE - config.DEFASAGEM_PUBLICACAO_PIB,
    }
    anos = {c: sorted(df[c].dropna().unique().tolist()) for c in PROCEDENCIA_IBGE}
    posteriores = {c: [a for a in v if a > tetos[c]] for c, v in anos.items()}
    if any(posteriores.values()):
        raise ValueError(
            f"IBGE com ano além do publicado na data da previsão: {posteriores} "
            f"(tetos {tetos}). Regere com enriquecimento_ibge.py."
        )
    logger.info("Enriquecimento IBGE — anos de referência: %s", anos)
    return df.drop(columns=PROCEDENCIA_IBGE)


def construir() -> pd.DataFrame:
    """Base analítica: uma linha por aluno do ano-alvo, com contexto anterior."""
    config.ensure_dirs()

    alunos = _alunos(config.ANO_ALVO, [
        "id_aluno", "id_escola", "id_municipio", "sigla_uf", "rede_descricao",
        "caderno", "alfabetizado", "peso_aluno", "registro_valido", "serie",
    ]).copy()
    logger.info("Alunos válidos em %d: %d", config.ANO_ALVO, len(alunos))

    base = alunos[[
        "id_aluno", "id_escola", "id_municipio", "sigla_uf", "rede_descricao",
        "caderno", "peso_aluno",
    ]].copy()
    base["alfabetizado"] = (alunos["alfabetizado"] == config.ALFABETIZADO_SIM).astype(int)
    base["regiao"] = base["sigla_uf"].map(REGIAO)

    base = base.merge(contexto_municipio(), on=["id_municipio", "rede_descricao"], how="left")
    base = base.merge(contexto_uf(), on=["sigla_uf", "rede_descricao"], how="left")
    base = base.merge(metas_municipais(), on=["id_municipio", "rede_descricao"], how="left")

    base = base.merge(metas_estaduais(), on=["sigla_uf", "rede_descricao"], how="left")

    externo = enriquecimento_externo()
    if externo is not None:
        base = base.merge(externo, on="id_municipio", how="left")
        logger.info("Enriquecimento IBGE aplicado: %s", [c for c in externo.columns if c != "id_municipio"])

    # distância entre a meta pactuada e o ponto de partida do município
    base["dist_meta_partida"] = (base["meta_do_ano"] - base["mun_taxa_ano_anterior"]).round(2)
    # o quanto a meta municipal é mais (ou menos) ambiciosa que a do estado
    base["dist_meta_mun_uf"] = (base["meta_do_ano"] - base["meta_uf_do_ano"]).round(2)

    nacional = meta_nacional()
    if nacional is not None:
        base["dist_meta_mun_nacional"] = (base["meta_do_ano"] - nacional).round(2)
        logger.info("Meta nacional da rede Pública em %d: %.2f", config.ANO_ALVO, nacional)
    else:
        logger.warning("Meta nacional ausente na Gold — dist_meta_mun_nacional não criada")

    cobertura = base.notna().mean().sort_values()
    logger.info("Cobertura das features (3 menores): %s",
                cobertura.head(3).round(3).to_dict())
    logger.info("Base analítica: %d linhas x %d colunas", len(base), base.shape[1])
    return base.reset_index(drop=True)


def amostrar(base: pd.DataFrame, n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Amostra estratificada pelo alvo, preservando sua proporção."""
    n = n or config.TAMANHO_AMOSTRA
    seed = seed if seed is not None else config.SEED
    if len(base) <= n:
        return base
    fracao = n / len(base)
    partes = [
        grupo.sample(frac=fracao, random_state=seed)
        for _, grupo in base.groupby("alfabetizado")
    ]
    amostra = pd.concat(partes).sample(frac=1, random_state=seed)
    logger.info("Amostra: %d linhas (proporção do alvo preservada)", len(amostra))
    return amostra.reset_index(drop=True)


if __name__ == "__main__":
    df = construir()
    destino = config.DATA_DIR / "base_analitica.parquet"
    df.to_parquet(destino, index=False)
    logger.info("Base salva em %s", destino)
