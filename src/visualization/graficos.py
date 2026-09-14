"""Gráficos da análise exploratória e da avaliação."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

COR = "#1E2761"
COR2 = "#D98E04"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def _salvar(fig, nome: str) -> str:
    config.ensure_dirs()
    caminho = config.IMAGES_DIR / f"{nome}.png"
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    return str(caminho)


def distribuicao_alvo(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    contagem = df["alfabetizado"].value_counts(normalize=True).sort_index() * 100
    ax.bar(["Não alfabetizado", "Alfabetizado"], contagem.values, color=[COR2, COR])
    for i, v in enumerate(contagem.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("% dos alunos"); ax.set_ylim(0, 75)
    ax.set_title("Distribuição do alvo — alunos avaliados em 2024")
    return _salvar(fig, "01_distribuicao_alvo")


def taxa_por_categoria(df: pd.DataFrame, coluna: str, titulo: str, nome: str) -> str:
    taxa = df.groupby(coluna)["alfabetizado"].agg(["mean", "size"])
    taxa = taxa[taxa["size"] > 500].sort_values("mean")
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.28 * len(taxa))))
    ax.barh(taxa.index.astype(str), taxa["mean"] * 100, color=COR)
    ax.axvline(df["alfabetizado"].mean() * 100, color=COR2, ls="--", lw=1.2,
               label="média nacional")
    ax.set_xlabel("% alfabetizados"); ax.set_title(titulo); ax.legend()
    return _salvar(fig, nome)


def distribuicoes_numericas(df: pd.DataFrame, colunas: list[str]) -> str:
    colunas = [c for c in colunas if c in df.columns][:6]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, col in zip(axes.ravel(), colunas):
        dados = df[col].dropna()
        ax.hist(dados, bins=40, color=COR, alpha=0.85)
        ax.set_title(col, fontsize=8)
    for ax in axes.ravel()[len(colunas):]:
        ax.axis("off")
    fig.suptitle("Distribuições das principais variáveis numéricas", y=1.0)
    return _salvar(fig, "02_distribuicoes")


def correlacoes(df: pd.DataFrame, colunas: list[str]) -> str:
    colunas = [c for c in colunas if c in df.columns]
    corr = df[colunas + ["alfabetizado"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlação entre variáveis e o alvo")
    ax.grid(False)
    return _salvar(fig, "03_correlacoes")


def relacao_contexto_alvo(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, rotulo in [
        (axes[0], "mun_taxa_ano_anterior", "Taxa do município em 2023 (%)"),
        (axes[1], "uf_taxa_ano_anterior", "Taxa da UF em 2023 (%)"),
    ]:
        sub = df[[col, "alfabetizado"]].dropna()
        faixas = pd.cut(sub[col], bins=np.arange(0, 101, 10))
        taxa = sub.groupby(faixas, observed=True)["alfabetizado"].mean() * 100
        ax.plot([i.mid for i in taxa.index], taxa.values, "o-", color=COR)
        ax.set_xlabel(rotulo); ax.set_ylabel("% alfabetizados em 2024")
        ax.set_ylim(0, 100)
    fig.suptitle("Contexto do ano anterior x resultado do ano seguinte")
    return _salvar(fig, "04_contexto_vs_alvo")


def curvas_desempenho(y, prob, peso=None) -> str:
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    fpr, tpr, _ = roc_curve(y, prob, sample_weight=peso)
    axes[0].plot(fpr, tpr, color=COR, lw=2,
                 label=f"AUC = {roc_auc_score(y, prob, sample_weight=peso):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_xlabel("Falsos positivos"); axes[0].set_ylabel("Verdadeiros positivos")
    axes[0].set_title("Curva ROC"); axes[0].legend()

    prec, rec, _ = precision_recall_curve(y, prob, sample_weight=peso)
    axes[1].plot(rec, prec, color=COR, lw=2,
                 label=f"AP = {average_precision_score(y, prob, sample_weight=peso):.3f}")
    axes[1].axhline(np.average(y, weights=peso), ls="--", color="gray", lw=1, label="prevalência")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precisão")
    axes[1].set_title("Curva Precisão-Recall"); axes[1].legend()
    return _salvar(fig, "08_curvas_desempenho")


def matriz_confusao(matriz) -> str:
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(matriz, cmap="Blues")
    rotulos = ["Não alfabetizado", "Alfabetizado"]
    ax.set_xticks([0, 1], rotulos, fontsize=8); ax.set_yticks([0, 1], rotulos, fontsize=8)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    total = matriz.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matriz[i, j]:,}\n({matriz[i, j]/total:.1%})".replace(",", "."),
                    ha="center", va="center",
                    color="white" if matriz[i, j] > total / 3 else "black", fontsize=8)
    ax.set_title("Matriz de confusão (teste)"); ax.grid(False)
    return _salvar(fig, "09_matriz_confusao")


def calibracao(y, prob) -> str:
    from sklearn.calibration import calibration_curve
    real, previsto = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="calibração perfeita")
    ax.plot(previsto, real, "o-", color=COR, label="modelo")
    ax.set_xlabel("Probabilidade prevista"); ax.set_ylabel("Frequência observada")
    ax.set_title("Calibração"); ax.legend()
    return _salvar(fig, "10_calibracao")


def importancia(nomes, valores, titulo: str, nome: str, n: int = 15) -> str:
    ordem = np.argsort(valores)[-n:]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(ordem) + 1.2))
    ax.barh([nomes[i] for i in ordem], [valores[i] for i in ordem], color=COR)
    ax.set_title(titulo); ax.set_xlabel("importância")
    return _salvar(fig, nome)


def shap_resumo(valores, X: pd.DataFrame) -> str:
    """Beeswarm: importância e direção do efeito de cada variável."""
    import shap
    plt.figure(figsize=(7, 6))
    shap.summary_plot(valores, X, max_display=15, show=False)
    plt.title("SHAP — efeito por variável")
    # o shap desenha na figura corrente; é ela que precisa ser salva
    return _salvar(plt.gcf(), "14_shap_resumo")


def shap_dependencia(valores, X: pd.DataFrame, variavel: str) -> str:
    """Efeito detalhado de uma variável, com a interação mais forte na cor."""
    import shap
    fig, ax = plt.subplots(figsize=(5.5, 4))
    shap.dependence_plot(variavel, valores, X, ax=ax, show=False)
    return _salvar(fig, "15_shap_dependencia")


def escolha_k(ks, inercias, silhuetas) -> str:
    """Cotovelo e silhueta lado a lado, para justificar o número de perfis."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(ks, inercias, "o-", color=COR)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inércia (WCSS)")
    axes[0].set_title("Método do cotovelo")
    axes[1].plot(ks, silhuetas, "o-", color=COR2)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhueta média")
    axes[1].set_title("Índice de silhueta")
    return _salvar(fig, "16_escolha_k")


def curva_aprendizado(df: pd.DataFrame) -> str:
    """Diagnóstico viés-variância: mais dados ajudariam ou o teto é o dado?"""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(df["n_treino"], df["auc_treino"], "o-", color=COR, label="treino")
    ax.plot(df["n_treino"], df["auc_validacao"], "o-", color=COR2, label="validação")
    ax.set_xlabel("alunos no treino"); ax.set_ylabel("ROC AUC")
    ax.set_title("Curva de aprendizado"); ax.legend()
    return _salvar(fig, "17_curva_aprendizado")


def curva_validacao(df: pd.DataFrame) -> str:
    """Erro em função da complexidade: onde está o ponto ideal."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(df["max_leaf_nodes"], df["auc_treino"], "o-", color=COR, label="treino")
    ax.plot(df["max_leaf_nodes"], df["auc_validacao"], "o-", color=COR2, label="validação")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("max_leaf_nodes (complexidade)"); ax.set_ylabel("ROC AUC")
    ax.set_title("Curva de validação"); ax.legend()
    return _salvar(fig, "18_curva_validacao")


def perfis_pca(componentes, rotulos, variancia: float) -> str:
    """Projeção 2D dos municípios, colorida pelo perfil."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for nome in pd.unique(rotulos):
        m = rotulos == nome
        ax.scatter(componentes[m, 0], componentes[m, 1], s=8, alpha=0.6, label=nome)
    ax.set_xlabel("componente 1"); ax.set_ylabel("componente 2")
    ax.set_title(f"Perfis municipais em 2D (PCA — {variancia:.0%} da variância)")
    ax.legend(markerscale=2, fontsize=8)
    return _salvar(fig, "19_perfis_pca")


def risco_municipal(df, coluna="risco", n: int = 15) -> str:
    top = df.nlargest(n, coluna)
    rotulos = [f"{r.nome_municipio}/{r.sigla_uf}" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * n + 1))
    ax.barh(rotulos[::-1], top[coluna].values[::-1] * 100, color=COR2)
    ax.set_xlabel("% de alunos com previsão de não alfabetização")
    ax.set_title(f"{n} municípios com maior risco previsto")
    return _salvar(fig, "12_municipios_risco")
