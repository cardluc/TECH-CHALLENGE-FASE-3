"""Treinamento, validação e otimização dos modelos.

Separação sem vazamento: o split é feito por MUNICÍPIO (GroupShuffleSplit).
Como as variáveis descrevem o contexto municipal, deixar o mesmo município
nos dois lados permitiria ao modelo memorizar o contexto em vez de
generalizar. O teste mede o que interessa: desempenho em municípios que o
modelo nunca viu.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, RandomizedSearchCV,
                                     cross_validate, learning_curve, validation_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.preprocessing.pipeline import construir_preprocessador
from src.utils import get_logger
from src.visualization import graficos

logger = get_logger("treino")

ALVO = "alfabetizado"
GRUPO = "id_municipio"


def separar_treino_teste(base: pd.DataFrame):
    """Split agrupado por município, estratificação natural preservada."""
    divisor = GroupShuffleSplit(
        n_splits=1, test_size=config.TESTE_PROPORCAO, random_state=config.SEED
    )
    idx_treino, idx_teste = next(divisor.split(base, groups=base[GRUPO]))
    treino, teste = base.iloc[idx_treino], base.iloc[idx_teste]
    logger.info(
        "Treino: %d alunos / %d municípios | Teste: %d alunos / %d municípios",
        len(treino), treino[GRUPO].nunique(), len(teste), teste[GRUPO].nunique(),
    )
    assert not set(treino[GRUPO]) & set(teste[GRUPO]), "vazamento: município nos dois conjuntos"
    return treino, teste


def _pipe(X: pd.DataFrame, estimador, escalador: str = "padrao") -> Pipeline:
    """Pré-processamento e modelo num objeto só, reajustado a cada dobra."""
    return Pipeline([
        ("preprocessamento", construir_preprocessador(X, escalador)),
        ("modelo", estimador),
    ])


def pesos(df: pd.DataFrame) -> dict:
    """Peso amostral roteado para o passo `modelo` do Pipeline.

    O scorer da validação continua sem peso: ali o objetivo é ranquear
    algoritmos ajustados em condições idênticas, não estimar a taxa
    populacional. As métricas ponderadas ficam em `avaliar.py`.
    """
    return {"modelo__sample_weight": df["peso_aluno"].to_numpy()}


def modelos_candidatos() -> dict[str, object]:
    """Uma representante de cada família de classificação: linear probabilística,
    árvore, ensemble (bagging e boosting), margem máxima e bayesiana."""
    return {
        "baseline": DummyClassifier(strategy="prior"),
        "regressao_logistica": LogisticRegression(max_iter=1000, random_state=config.SEED),
        "arvore_decisao": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, random_state=config.SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=50, n_jobs=-1, random_state=config.SEED
        ),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=config.SEED),
        "svm_linear": LinearSVC(C=0.1, dual="auto", random_state=config.SEED),
        "naive_bayes": GaussianNB(),
    }


def comparar(treino: pd.DataFrame, escalador: str = "padrao") -> pd.DataFrame:
    """Validação cruzada agrupada para escolher o algoritmo.

    `return_train_score` permite o diagnóstico viés-variância: erro de treino
    próximo do de validação e ambos altos indica subajuste; distantes, sobreajuste.
    """
    X = treino.drop(columns=[ALVO])
    y = treino[ALVO]
    grupos = treino[GRUPO]
    cv = GroupKFold(n_splits=config.N_FOLDS)

    linhas = []
    for nome, estimador in modelos_candidatos().items():
        scores = cross_validate(
            _pipe(X, estimador, escalador), X, y, groups=grupos, cv=cv,
            scoring=["roc_auc", "accuracy", "f1", "average_precision"],
            return_train_score=True, n_jobs=1, params=pesos(treino),
        )
        treino_auc = scores["train_roc_auc"].mean()
        val_auc = scores["test_roc_auc"].mean()
        linhas.append({
            "modelo": nome,
            "roc_auc": round(val_auc, 4),
            "roc_auc_dp": scores["test_roc_auc"].std().round(4),
            "roc_auc_treino": round(treino_auc, 4),
            "gap_treino_validacao": round(treino_auc - val_auc, 4),
            "acuracia": scores["test_accuracy"].mean().round(4),
            "f1": scores["test_f1"].mean().round(4),
            "pr_auc": scores["test_average_precision"].mean().round(4),
            "tempo_s": scores["fit_time"].mean().round(1),
        })
        logger.info("%-22s ROC AUC %.4f (±%.4f) | gap treino %.4f", nome,
                    linhas[-1]["roc_auc"], linhas[-1]["roc_auc_dp"],
                    linhas[-1]["gap_treino_validacao"])
    return pd.DataFrame(linhas).sort_values("roc_auc", ascending=False).reset_index(drop=True)


VARIAVEIS_IBGE = ["populacao", "pib", "pib_per_capita"]


def ablacao_ibge(treino: pd.DataFrame, com_ibge: pd.DataFrame) -> pd.DataFrame:
    """Compara os mesmos algoritmos com e sem as variáveis socioeconômicas.

    Roda na mesma amostra, mesmo split e mesmas dobras — a única diferença são
    as colunas do IBGE. Sem isto, a afirmação de que o enriquecimento externo
    não altera o resultado ficaria sem evidência reproduzível no repositório.
    """
    presentes = [c for c in VARIAVEIS_IBGE if c in treino.columns]
    if not presentes:
        logger.warning("Ablação do IBGE ignorada: a base não tem %s", VARIAVEIS_IBGE)
        return pd.DataFrame()

    sem = comparar(treino.drop(columns=presentes))
    juntos = (
        com_ibge[["modelo", "roc_auc"]].rename(columns={"roc_auc": "com_ibge"})
        .merge(sem[["modelo", "roc_auc"]].rename(columns={"roc_auc": "sem_ibge"}),
               on="modelo")
    )
    juntos["diferenca"] = (juntos["com_ibge"] - juntos["sem_ibge"]).round(4)
    logger.info("Ablação IBGE — diferença de ROC AUC entre %+.4f e %+.4f",
                juntos["diferenca"].min(), juntos["diferenca"].max())
    return juntos.sort_values("com_ibge", ascending=False).reset_index(drop=True)


def curva_aprendizado(treino: pd.DataFrame, n: int = 40_000) -> pd.DataFrame:
    """Erro de treino e validação em função do volume de dados.

    Responde empiricamente se o teto do modelo é falta de dados (curvas ainda
    se aproximando) ou falta de informação nas variáveis (curvas já paradas).
    """
    sub = treino.sample(min(n, len(treino)), random_state=config.SEED)
    X, y = sub.drop(columns=[ALVO]), sub[ALVO]
    tamanhos, treino_sc, val_sc = learning_curve(
        _pipe(X, HistGradientBoostingClassifier(random_state=config.SEED)),
        X, y, groups=sub[GRUPO], cv=GroupKFold(n_splits=3),
        train_sizes=np.linspace(0.1, 1.0, 6), scoring="roc_auc", n_jobs=1,
        params=pesos(sub),
    )
    df = pd.DataFrame({
        "n_treino": tamanhos,
        "auc_treino": treino_sc.mean(axis=1).round(4),
        "auc_validacao": val_sc.mean(axis=1).round(4),
    })
    df["gap"] = (df["auc_treino"] - df["auc_validacao"]).round(4)
    graficos.curva_aprendizado(df)
    # o passo final não é uma duplicação do volume; o percentual real entra no log
    cresc = 100 * (df["n_treino"].iloc[-1] / df["n_treino"].iloc[-2] - 1)
    logger.info(
        "Curva de aprendizado — %+.0f%% de dados no último passo (%d → %d) rende %+.4f de AUC",
        cresc, df["n_treino"].iloc[-2], df["n_treino"].iloc[-1],
        df["auc_validacao"].iloc[-1] - df["auc_validacao"].iloc[-2],
    )
    return df


def curva_validacao(treino: pd.DataFrame, n: int = 40_000) -> pd.DataFrame:
    """Erro em função da complexidade do modelo (max_leaf_nodes)."""
    sub = treino.sample(min(n, len(treino)), random_state=config.SEED)
    X, y = sub.drop(columns=[ALVO]), sub[ALVO]
    valores = [4, 8, 15, 31, 63, 127]
    treino_sc, val_sc = validation_curve(
        _pipe(X, HistGradientBoostingClassifier(random_state=config.SEED)),
        X, y, groups=sub[GRUPO], cv=GroupKFold(n_splits=3),
        param_name="modelo__max_leaf_nodes", param_range=valores,
        scoring="roc_auc", n_jobs=1, params=pesos(sub),
    )
    df = pd.DataFrame({
        "max_leaf_nodes": valores,
        "auc_treino": treino_sc.mean(axis=1).round(4),
        "auc_validacao": val_sc.mean(axis=1).round(4),
    })
    graficos.curva_validacao(df)
    logger.info("Complexidade ótima (max_leaf_nodes): %d",
                df.loc[df["auc_validacao"].idxmax(), "max_leaf_nodes"])
    return df


ESPACO_BUSCA = {
    "modelo__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "modelo__max_iter": [150, 300, 500],
    # 4 e 8 incluídos porque a curva de validação aponta o ótimo abaixo de 15
    "modelo__max_leaf_nodes": [4, 8, 15, 31, 63],
    "modelo__min_samples_leaf": [20, 50, 100],
    "modelo__l2_regularization": [0.0, 0.1, 1.0],
}


def otimizar(treino: pd.DataFrame, n_iter: int = 10) -> RandomizedSearchCV:
    """Busca aleatória de hiperparâmetros com validação cruzada agrupada.

    `refit=False` porque o ajuste final é refeito em `treinar()`; reajustar aqui
    custaria um treino inteiro que seria descartado.
    """
    X = treino.drop(columns=[ALVO])
    busca = RandomizedSearchCV(
        _pipe(X, HistGradientBoostingClassifier(random_state=config.SEED)),
        ESPACO_BUSCA, n_iter=n_iter, scoring="roc_auc",
        cv=GroupKFold(n_splits=3), random_state=config.SEED, n_jobs=1, refit=False,
    )
    busca.fit(X, treino[ALVO], groups=treino[GRUPO], **pesos(treino))
    logger.info("Melhor ROC AUC na validação: %.4f", busca.best_score_)
    logger.info("Hiperparâmetros: %s", busca.best_params_)
    return busca


def treinar(base: pd.DataFrame) -> dict:
    config.ensure_dirs()
    treino, teste = separar_treino_teste(base)

    comparacao = comparar(treino)
    comparacao.to_csv(config.REPORTS_DIR / "comparacao_modelos.csv", index=False)

    # o escalonamento robusto é testado porque populacao e pib são assimétricos
    robusto = comparar(treino, escalador="robusto")
    robusto.to_csv(config.REPORTS_DIR / "comparacao_escalador.csv", index=False)

    ablacao = ablacao_ibge(treino, comparacao)
    ablacao.to_csv(config.REPORTS_DIR / "ablacao_ibge.csv", index=False)

    aprendizado = curva_aprendizado(treino)
    aprendizado.to_csv(config.REPORTS_DIR / "curva_aprendizado.csv", index=False)
    validacao = curva_validacao(treino)
    validacao.to_csv(config.REPORTS_DIR / "curva_validacao.csv", index=False)

    busca = otimizar(treino)

    X = treino.drop(columns=[ALVO])
    modelo = _pipe(X, HistGradientBoostingClassifier(random_state=config.SEED))
    modelo.set_params(**busca.best_params_)
    modelo.fit(X, treino[ALVO], **pesos(treino))

    joblib.dump(modelo, config.MODELS_DIR / "modelo.joblib")
    treino.to_parquet(config.DATA_DIR / "treino.parquet", index=False)
    teste.to_parquet(config.DATA_DIR / "teste.parquet", index=False)

    resumo = {
        "comparacao": comparacao.to_dict(orient="records"),
        "comparacao_escalador_robusto": robusto.to_dict(orient="records"),
        "ablacao_ibge": ablacao.to_dict(orient="records"),
        "curva_aprendizado": aprendizado.to_dict(orient="records"),
        "curva_validacao": validacao.to_dict(orient="records"),
        "melhores_parametros": {k: v for k, v in busca.best_params_.items()},
        "roc_auc_validacao": round(float(busca.best_score_), 4),
        "n_treino": int(len(treino)),
        "n_teste": int(len(teste)),
        "municipios_treino": int(treino[GRUPO].nunique()),
        "municipios_teste": int(teste[GRUPO].nunique()),
    }
    (config.REPORTS_DIR / "treino.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Modelo salvo em %s", config.MODELS_DIR / "modelo.joblib")
    return resumo


if __name__ == "__main__":
    from src.preprocessing.base_analitica import amostrar, construir
    treinar(amostrar(construir()))
