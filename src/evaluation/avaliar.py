"""Avaliação no conjunto de teste e interpretabilidade do modelo."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from src import config
from src.modeling.treinar import ALVO
from src.utils import get_logger
from src.visualization import graficos

logger = get_logger("avaliacao")


def carregar():
    modelo = joblib.load(config.MODELS_DIR / "modelo.joblib")
    teste = pd.read_parquet(config.DATA_DIR / "teste.parquet")
    return modelo, teste


def metricas(y, prob, peso, limiar: float = 0.5) -> dict:
    """Métricas globais e por classe.

    A classe positiva do modelo é `alfabetizado=1`. Para a política pública,
    porém, a classe de interesse é a NEGATIVA (não alfabetizado): é ela que
    precisa ser sinalizada. Por isso as duas são reportadas separadamente.
    """
    pred = (prob >= limiar).astype(int)
    m = {
        "limiar": limiar,
        "roc_auc": round(roc_auc_score(y, prob, sample_weight=peso), 4),
        "pr_auc": round(average_precision_score(y, prob, sample_weight=peso), 4),
        "acuracia": round(accuracy_score(y, pred, sample_weight=peso), 4),
        "brier": round(brier_score_loss(y, prob, sample_weight=peso), 4),
    }
    for rotulo, classe in (("alfabetizado", 1), ("nao_alfabetizado", 0)):
        m[f"precisao_{rotulo}"] = round(
            precision_score(y, pred, pos_label=classe, sample_weight=peso), 4)
        m[f"recall_{rotulo}"] = round(
            recall_score(y, pred, pos_label=classe, sample_weight=peso), 4)
        m[f"f1_{rotulo}"] = round(
            f1_score(y, pred, pos_label=classe, sample_weight=peso), 4)
    return m


def escolher_limiar(modelo, treino: pd.DataFrame, recall_alvo: float = 0.7) -> float:
    """Menor limiar que atinge o recall desejado para NÃO alfabetizado.

    Escolhido na validação agrupada do treino; o teste permanece reservado.
    """
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from src.modeling.treinar import GRUPO, pesos

    X = treino.drop(columns=[ALVO])
    prob = cross_val_predict(modelo, X, treino[ALVO], groups=treino[GRUPO],
                             cv=GroupKFold(n_splits=3), method="predict_proba",
                             n_jobs=1, params=pesos(treino))[:, 1]
    y = treino[ALVO].to_numpy()
    # o recall é ponderado, como as métricas de teste: o limiar precisa ser
    # escolhido na mesma escala em que será cobrado
    peso = treino["peso_aluno"].to_numpy()
    for limiar in np.arange(0.30, 0.86, 0.01):
        pred = (prob >= limiar).astype(int)
        if recall_score(y, pred, pos_label=0, sample_weight=peso) >= recall_alvo:
            logger.info("Limiar escolhido na validação: %.2f (recall alvo %.2f)",
                        limiar, recall_alvo)
            return round(float(limiar), 2)
    logger.warning("Recall alvo %.2f inatingível; mantendo 0.5", recall_alvo)
    return 0.5


def importancia_permutacao(modelo, teste: pd.DataFrame, n: int = 20_000) -> pd.DataFrame:
    """Quanto o desempenho cai ao embaralhar cada variável."""
    amostra = teste.sample(min(n, len(teste)), random_state=config.SEED)
    X = amostra.drop(columns=[ALVO])
    resultado = permutation_importance(
        modelo, X, amostra[ALVO], n_repeats=5, random_state=config.SEED,
        scoring="roc_auc", n_jobs=1, sample_weight=amostra["peso_aluno"].to_numpy(),
    )
    return (
        pd.DataFrame({"variavel": X.columns, "importancia": resultado.importances_mean.round(5)})
        .sort_values("importancia", ascending=False).reset_index(drop=True)
    )


def shap_valores(modelo, teste: pd.DataFrame):
    """SHAP sobre uma subamostra: ranking, valores brutos e matriz transformada."""
    import shap

    amostra = teste.sample(min(config.AMOSTRA_SHAP, len(teste)), random_state=config.SEED)
    X = amostra.drop(columns=[ALVO])
    pre = modelo.named_steps["preprocessamento"]
    nomes = list(pre.get_feature_names_out())
    Xt = pd.DataFrame(pre.transform(X), columns=nomes)

    explicador = shap.TreeExplainer(modelo.named_steps["modelo"])
    valores = explicador.shap_values(Xt)
    if isinstance(valores, list):
        valores = valores[1]
    medias = np.abs(valores).mean(axis=0)
    ranking = (
        pd.DataFrame({"variavel": nomes, "shap_medio": medias.round(5)})
        .sort_values("shap_medio", ascending=False).reset_index(drop=True)
    )
    return ranking, valores, Xt


def executar() -> dict:
    config.ensure_dirs()
    modelo, teste = carregar()
    y = teste[ALVO].to_numpy()
    peso = teste["peso_aluno"].to_numpy()
    prob = modelo.predict_proba(teste.drop(columns=[ALVO]))[:, 1]

    treino = pd.read_parquet(config.DATA_DIR / "treino.parquet")
    limiar = escolher_limiar(modelo, treino)

    m = metricas(y, prob, peso, limiar)
    logger.info("Teste — ROC AUC %.4f | recall não alfabetizado %.4f | precisão %.4f",
                m["roc_auc"], m["recall_nao_alfabetizado"], m["precisao_nao_alfabetizado"])

    graficos.curvas_desempenho(y, prob, peso)
    graficos.matriz_confusao(confusion_matrix(y, (prob >= limiar).astype(int)))
    graficos.calibracao(y, prob)

    perm = importancia_permutacao(modelo, teste)
    graficos.importancia(perm["variavel"].tolist(), perm["importancia"].to_numpy(),
                         "Importância por permutação (queda no ROC AUC)", "11_importancia_permutacao")
    logger.info("Top 3 por permutação: %s", perm.head(3)["variavel"].tolist())

    shap_df, shap_vals, shap_X = shap_valores(modelo, teste)
    graficos.importancia(shap_df["variavel"].tolist(), shap_df["shap_medio"].to_numpy(),
                         "SHAP — contribuição média absoluta", "13_shap")
    graficos.shap_resumo(shap_vals, shap_X)
    graficos.shap_dependencia(shap_vals, shap_X, shap_df.loc[0, "variavel"])
    logger.info("Top 3 por SHAP: %s", shap_df.head(3)["variavel"].tolist())

    perm.to_csv(config.REPORTS_DIR / "importancia_permutacao.csv", index=False)
    shap_df.to_csv(config.REPORTS_DIR / "importancia_shap.csv", index=False)
    (config.REPORTS_DIR / "metricas.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metricas": m, "permutacao": perm, "shap": shap_df}


if __name__ == "__main__":
    executar()
