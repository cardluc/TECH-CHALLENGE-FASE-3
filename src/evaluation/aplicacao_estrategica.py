"""Responde às perguntas de negócio a partir das previsões do modelo.

As previsões municipais são geradas *out-of-fold*: cada aluno é previsto
por um modelo que não viu o seu município no treino. Sem isso, o ranking
de risco seria otimista justamente nos municípios usados para treinar.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.modeling.treinar import ALVO, GRUPO
from src.utils import get_logger
from src.visualization import graficos

logger = get_logger("estrategia")

# o número de alunos fica de fora: as capitais são ordens de grandeza maiores
# e dominariam o agrupamento, separando por porte em vez de perfil educacional
COLUNAS_PERFIL = ["indicador_previsto", "taxa_ano_anterior", "meta_do_ano", "gap_previsto_meta"]
K_PERFIS = 4


def previsoes_out_of_fold(modelo, base: pd.DataFrame, n_splits: int = 3) -> np.ndarray:
    X = base.drop(columns=[ALVO])
    from src.modeling.treinar import pesos

    return cross_val_predict(
        modelo, X, base[ALVO], groups=base[GRUPO],
        cv=GroupKFold(n_splits=n_splits), method="predict_proba", n_jobs=1,
        params=pesos(base),
    )[:, 1]


def risco_por_municipio(base: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """Agrega por MUNICÍPIO × REDE.

    A meta e o histórico são pactuados por rede; agregar só por município
    compararia uma previsão que mistura redes com uma meta de uma rede só.
    As médias usam o peso amostral, como no ajuste e na avaliação do modelo.
    """
    df = base.copy()
    df["prob_alfabetizado"] = prob

    def ponderada(g, coluna):
        return round(100 * np.average(g[coluna], weights=g["peso_aluno"]), 2)

    chaves = ["id_municipio", "sigla_uf", "regiao", "rede_descricao"]
    linhas = []
    for chave, g in df.groupby(chaves, observed=True):
        linhas.append(dict(zip(chaves, chave), **{
            "alunos": len(g),
            "indicador_previsto": ponderada(g, "prob_alfabetizado"),
            "indicador_real": ponderada(g, ALVO),
            "meta_do_ano": g["meta_do_ano"].iloc[0],
            "taxa_ano_anterior": g["mun_taxa_ano_anterior"].iloc[0],
        }))
    agg = pd.DataFrame(linhas)

    agg["risco"] = 1 - agg["indicador_previsto"] / 100
    agg["gap_previsto_meta"] = (agg["indicador_previsto"] - agg["meta_do_ano"]).round(2)
    # sem meta pactuada não é possível dizer se há risco de descumprimento
    agg["sem_meta"] = agg["meta_do_ano"].isna()
    agg["risco_de_nao_atingir_meta"] = (~agg["sem_meta"]) & (agg["gap_previsto_meta"] < 0)
    return agg[agg["alunos"] >= 20].reset_index(drop=True)


def _padronizar(municipios: pd.DataFrame) -> np.ndarray:
    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ]).fit_transform(municipios[COLUNAS_PERFIL])


def diagnostico_k(municipios: pd.DataFrame, ks=range(2, 9)) -> pd.DataFrame:
    """Cotovelo, silhueta e comparação entre três famílias de clusterização.

    Sem rótulos não existe acurácia: a qualidade é medida pela coesão interna
    e separação entre grupos (silhueta), combinada com a leitura de negócio.
    """
    X = _padronizar(municipios)
    linhas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(X)
        hier = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        gmm = GaussianMixture(n_components=k, random_state=config.SEED).fit_predict(X)
        linhas.append({
            "k": k,
            "inercia": round(float(km.inertia_), 1),
            "silhueta_kmeans": round(silhouette_score(X, km.labels_), 4),
            "silhueta_hierarquico": round(silhouette_score(X, hier), 4),
            "silhueta_gmm": round(silhouette_score(X, gmm), 4),
            # Davies-Bouldin: menor é melhor (dispersão interna / separação)
            "davies_bouldin_kmeans": round(davies_bouldin_score(X, km.labels_), 4),
        })
    df = pd.DataFrame(linhas)
    graficos.escolha_k(df["k"], df["inercia"], df["silhueta_kmeans"])
    df.to_csv(config.REPORTS_DIR / "escolha_k.csv", index=False)
    melhor = df.loc[df["silhueta_kmeans"].idxmax(), "k"]
    logger.info("Silhueta máxima em k=%d; k adotado=%d (perfis acionáveis)", melhor, K_PERFIS)
    return df


def agrupar_perfis(municipios: pd.DataFrame, k: int = K_PERFIS) -> pd.DataFrame:
    """Agrupa municípios por perfil educacional (k-means)."""
    X = _padronizar(municipios)
    municipios = municipios.copy()
    municipios["perfil"] = KMeans(
        n_clusters=k, n_init=10, random_state=config.SEED
    ).fit_predict(X)
    return municipios


def _rotular_perfis(perfis: pd.DataFrame) -> dict[int, str]:
    ordem = perfis.sort_values("indicador_previsto")["perfil"].tolist()
    nomes = ["Crítico", "Atenção", "Intermediário", "Consolidado"]
    return {p: nomes[min(i, len(nomes) - 1)] for i, p in enumerate(ordem)}


def executar(modelo, base: pd.DataFrame, dim_municipio: pd.DataFrame) -> dict:
    config.ensure_dirs()
    logger.info("Gerando previsões out-of-fold para %d alunos", len(base))
    prob = previsoes_out_of_fold(modelo, base)

    municipios = risco_por_municipio(base, prob)
    municipios = municipios.merge(dim_municipio, on="id_municipio", how="left", suffixes=("", "_dim"))
    diagnostico = diagnostico_k(municipios)
    municipios = agrupar_perfis(municipios)

    perfis = municipios.groupby("perfil").agg(
        municipios=("id_municipio", "size"),
        indicador_previsto=("indicador_previsto", "mean"),
        taxa_ano_anterior=("taxa_ano_anterior", "mean"),
        meta_do_ano=("meta_do_ano", "mean"),
        alunos_medio=("alunos", "mean"),
    ).round(1).reset_index()
    rotulos = _rotular_perfis(perfis)
    perfis["nome_perfil"] = perfis["perfil"].map(rotulos)
    municipios["nome_perfil"] = municipios["perfil"].map(rotulos)

    pca = PCA(n_components=2, random_state=config.SEED)
    graficos.perfis_pca(pca.fit_transform(_padronizar(municipios)),
                        municipios["nome_perfil"].to_numpy(),
                        float(pca.explained_variance_ratio_.sum()))
    graficos.risco_municipal(municipios)

    # o percentual só faz sentido sobre quem TEM meta pactuada
    avaliaveis = municipios[~municipios["sem_meta"]]
    em_risco = municipios[municipios["risco_de_nao_atingir_meta"]]
    resumo = {
        "unidades_municipio_rede": int(len(municipios)),
        "municipios_distintos": int(municipios["id_municipio"].nunique()),
        "sem_meta_pactuada": int(municipios["sem_meta"].sum()),
        "avaliaveis_contra_meta": int(len(avaliaveis)),
        "em_risco_de_nao_atingir_meta": int(len(em_risco)),
        "pct_em_risco": round(100 * len(em_risco) / max(len(avaliaveis), 1), 1),
        "risco_medio_por_regiao": municipios.groupby("regiao")["risco"]
            .mean().mul(100).round(1).sort_values(ascending=False).to_dict(),
        "k_adotado": K_PERFIS,
        "silhueta_k_adotado": float(
            diagnostico.loc[diagnostico["k"] == K_PERFIS, "silhueta_kmeans"].iloc[0]),
        "perfis": perfis.to_dict(orient="records"),
    }

    # o produto final é a taxa municipal prevista: mede-se o erro dela, em p.p.,
    # contra a regra ingênua "repetir o indicador do ano anterior"
    comp = municipios.dropna(subset=["taxa_ano_anterior"])
    erro_modelo = (comp["indicador_previsto"] - comp["indicador_real"]).abs()
    erro_ingenuo = (comp["taxa_ano_anterior"] - comp["indicador_real"]).abs()
    resumo["erro_taxa_municipal_pp"] = {
        "mae_modelo": round(float(erro_modelo.mean()), 2),
        "rmse_modelo": round(float(np.sqrt((erro_modelo ** 2).mean())), 2),
        "mae_regra_ano_anterior": round(float(erro_ingenuo.mean()), 2),
        "n": int(len(comp)),
    }

    municipios.to_csv(config.REPORTS_DIR / "risco_municipal.csv", index=False)
    (config.REPORTS_DIR / "aplicacao_estrategica.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("%d de %d unidades com meta em risco (%.1f%%) | MAE da taxa: %.2f p.p. "
                "contra %.2f p.p. da regra do ano anterior",
                len(em_risco), len(avaliaveis), resumo["pct_em_risco"],
                resumo["erro_taxa_municipal_pp"]["mae_modelo"],
                resumo["erro_taxa_municipal_pp"]["mae_regra_ano_anterior"])
    return {"municipios": municipios, "perfis": perfis, "resumo": resumo}


if __name__ == "__main__":
    import joblib

    from src.preprocessing.base_analitica import construir

    modelo = joblib.load(config.MODELS_DIR / "modelo.joblib")
    # ranking sobre a base COMPLETA: o objetivo aqui é operacional
    # (priorizar municípios), não medir desempenho
    # amostra ampla: cobre praticamente todos os municípios mantendo o
    # tempo de execução razoável
    from src.preprocessing.base_analitica import amostrar
    base = amostrar(construir(), n=600_000)
    dim = pd.read_parquet(config.FASE2_DATA / "silver/dim_municipio")
    executar(modelo, base, dim)
