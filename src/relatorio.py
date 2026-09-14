"""Consolida os resultados da modelagem em reports/modelagem.md."""
from __future__ import annotations

import json

import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger("relatorio")


def _ler_json(nome: str) -> dict:
    return json.loads((config.REPORTS_DIR / nome).read_text(encoding="utf-8"))


def gerar() -> str:
    treino = _ler_json("treino.json")
    metricas = _ler_json("metricas.json")
    estrategia = _ler_json("aplicacao_estrategica.json")
    comparacao = pd.read_csv(config.REPORTS_DIR / "comparacao_modelos.csv")
    perm = pd.read_csv(config.REPORTS_DIR / "importancia_permutacao.csv")
    shap_df = pd.read_csv(config.REPORTS_DIR / "importancia_shap.csv")
    municipios = pd.read_csv(config.REPORTS_DIR / "risco_municipal.csv")

    L = ["# Modelagem — Predição de Alfabetização\n"]

    L.append("## 1. Desenho do experimento\n")
    L.append(
        f"- **Unidade**: um aluno avaliado em {config.ANO_ALVO} (2º ano do EF)\n"
        f"- **Alvo**: alfabetizado (proficiência ≥ 743 na escala Saeb)\n"
        f"- **Treino**: {treino['n_treino']:,} alunos em {treino['municipios_treino']:,} municípios\n"
        f"- **Teste**: {treino['n_teste']:,} alunos em {treino['municipios_teste']:,} municípios\n"
        "- **Separação**: por **município** (GroupShuffleSplit). Nenhum município "
        "aparece nos dois conjuntos — o teste mede generalização para municípios "
        "que o modelo nunca viu.\n".replace(",", ".")
    )

    L.append("## 2. Comparação de algoritmos\n")
    L.append("Validação cruzada agrupada por município, 5 dobras:\n")
    L.append(comparacao.to_markdown(index=False, floatfmt=".4f") + "\n")
    L.append(
        "O *baseline* (classe majoritária) confirma o piso de 0,5 de ROC AUC. As seis "
        "famílias de algoritmo ficam próximas, e o gap treino–validação é pequeno em "
        "todas: o limite é a **informação disponível**, não a capacidade do modelo nem "
        "sobreajuste.\n"
    )
    if treino.get("ablacao_ibge"):
        L.append("### Ablação: o enriquecimento socioeconômico serve?\n")
        ab = pd.DataFrame(treino["ablacao_ibge"])
        L.append(ab.to_markdown(index=False, floatfmt=".4f") + "\n")
        L.append(
            f"Mesma amostra, mesmo split e mesmas dobras; a única diferença são as "
            f"colunas do IBGE. As diferenças de ROC AUC ficam entre "
            f"{ab['diferenca'].min():+.4f} e {ab['diferenca'].max():+.4f}, abaixo do "
            "desvio entre dobras de qualquer um dos algoritmos. As variáveis foram "
            "mantidas porque a integração socioeconômica é requisito do projeto, mas "
            "o sinal que elas carregam já está nas variáveis de desempenho "
            "territorial.\n"
        )

    L.append("### Diagnóstico de viés e variância\n")
    aprendizado = pd.DataFrame(treino["curva_aprendizado"])
    ganho = aprendizado["auc_validacao"].iloc[-1] - aprendizado["auc_validacao"].iloc[-2]
    n0, n1 = int(aprendizado["n_treino"].iloc[-2]), int(aprendizado["n_treino"].iloc[-1])
    L.append(
        f"O último passo da curva acrescenta {100 * (n1 / n0 - 1):.0f}% de exemplos "
        f"({n0:,} → {n1:,}) e rende **{ganho:+.4f} de AUC**: o retorno marginal do "
        "volume já é desprezível *no intervalo avaliado*. Isso não demonstra que mais "
        "dados nunca ajudariam — exigiria estender a curva bem além deste limite. "
        "A curva de validação aponta o ótimo em "
        f"**{max(treino['curva_validacao'], key=lambda r: r['auc_validacao'])['max_leaf_nodes']} "
        "folhas**: o sinal é simples.\n".replace(",", ".")
    )
    L.append("![Curva de aprendizado](../images/17_curva_aprendizado.png)\n")
    L.append("![Curva de validação](../images/18_curva_validacao.png)\n")

    L.append("## 3. Hiperparâmetros e desempenho final\n")
    L.append(f"Melhores parâmetros (busca aleatória, 10 combinações):\n\n```\n"
             + json.dumps(treino["melhores_parametros"], indent=2) + "\n```\n")
    L.append(
        f"Desempenho no conjunto de teste, ponderado pelo peso amostral, no limiar "
        f"**{metricas['limiar']}** (escolhido na validação do treino com alvo de recall "
        "para a classe *não alfabetizado*):\n"
    )
    globais = ["roc_auc", "pr_auc", "acuracia", "brier"]
    L.append(pd.DataFrame([{k: metricas[k] for k in globais}])
             .to_markdown(index=False, floatfmt=".4f") + "\n")
    L.append("Por classe — a que interessa sinalizar é a **negativa**:\n")
    L.append(pd.DataFrame([
        {"classe": rotulo, "precisão": metricas[f"precisao_{rotulo}"],
         "recall": metricas[f"recall_{rotulo}"], "F1": metricas[f"f1_{rotulo}"]}
        for rotulo in ("alfabetizado", "nao_alfabetizado")
    ]).to_markdown(index=False, floatfmt=".4f") + "\n")
    L.append("![Curvas](../images/08_curvas_desempenho.png)\n")
    L.append("![Matriz de confusão](../images/09_matriz_confusao.png)\n")
    L.append("![Calibração](../images/10_calibracao.png)\n")
    L.append(
        f"O Brier score de {metricas['brier']:.4f} e a curva de calibração mostram que "
        "as probabilidades previstas são utilizáveis como **medida de risco**, não "
        "apenas como classificação binária.\n"
    )

    L.append("## 4. Interpretabilidade\n")
    L.append("### Importância por permutação (queda no ROC AUC)\n")
    L.append(perm.head(10).to_markdown(index=False, floatfmt=".5f") + "\n")
    L.append("![Permutação](../images/11_importancia_permutacao.png)\n")
    L.append("### SHAP (contribuição média absoluta)\n")
    L.append(shap_df.head(10).to_markdown(index=False, floatfmt=".5f") + "\n")
    L.append("![SHAP](../images/13_shap.png)\n")
    # derivado das tabelas acima: texto fixo aqui já contradisse o próprio ranking
    topo_perm, topo_shap = perm.iloc[0]["variavel"], shap_df.iloc[0]["variavel"]
    if topo_perm == topo_shap:
        L.append(
            f"As duas técnicas concordam no topo: **`{topo_shap}`** é a mais influente "
            f"pelos dois métodos, seguida de `{shap_df.iloc[1]['variavel']}`. "
            "A concordância entre métodos independentes reforça que o sinal é real e "
            "não artefato de um algoritmo específico.\n"
        )
    else:
        L.append(
            f"As técnicas divergem no topo: a permutação aponta `{topo_perm}` e o SHAP, "
            f"`{topo_shap}`. Divergência entre métodos independentes pede cautela na "
            "leitura do ranking.\n"
        )

    L.append("## 5. Aplicação estratégica\n")
    L.append(
        f"Previsões *out-of-fold* agregadas em **{estrategia['unidades_municipio_rede']:,} "
        f"unidades município × rede** ({estrategia['municipios_distintos']:,} municípios "
        "distintos). A agregação é por rede porque a meta é pactuada por rede; cada "
        "unidade é prevista por um modelo que não viu aquele município no treino.\n"
        .replace(",", ".")
    )
    L.append("### Risco de não atingir a meta\n")
    L.append(
        f"**{estrategia['em_risco_de_nao_atingir_meta']:,} de "
        f"{estrategia['avaliaveis_contra_meta']:,} unidades com meta pactuada "
        f"({estrategia['pct_em_risco']}%)** têm indicador previsto abaixo da meta de "
        f"{config.ANO_ALVO}. Outras {estrategia['sem_meta_pactuada']:,} unidades não "
        "têm meta e ficam classificadas como *não avaliáveis* — não como sem risco.\n"
        .replace(",", ".")
    )
    L.append("### Validação do produto final\n")
    erro = estrategia["erro_taxa_municipal_pp"]
    L.append(
        f"Erro da taxa municipal prevista: **MAE de {erro['mae_modelo']} p.p.** "
        f"(RMSE {erro['rmse_modelo']}), contra **{erro['mae_regra_ano_anterior']} p.p.** "
        f"da regra ingênua *repetir o indicador do ano anterior* — {erro['n']:,} unidades "
        "comparadas. É este número, e não o ROC AUC por aluno, que justifica usar o "
        "ranking para priorização.\n".replace(",", ".")
    )
    L.append("### Risco médio por região (%)\n")
    L.append(pd.Series(estrategia["risco_medio_por_regiao"], name="risco (%)")
             .rename_axis("região").reset_index().to_markdown(index=False, floatfmt=".1f") + "\n")
    L.append("### Perfis de município (k-means)\n")
    L.append(pd.DataFrame(estrategia["perfis"])[
        ["nome_perfil", "municipios", "indicador_previsto", "taxa_ano_anterior", "meta_do_ano"]
    ].to_markdown(index=False, floatfmt=".1f") + "\n")
    escolha_k = pd.read_csv(config.REPORTS_DIR / "escolha_k.csv")
    melhor_k = int(escolha_k.loc[escolha_k["silhueta_kmeans"].idxmax(), "k"])
    L.append(
        f"k adotado = **{estrategia['k_adotado']}** (silhueta {estrategia['silhueta_k_adotado']:.4f}). "
        f"A silhueta é máxima em k={melhor_k}; a escolha de {estrategia['k_adotado']} perfis é "
        "de negócio, não estatística — cada um corresponde a uma ação distinta"
        + (f", enquanto k={melhor_k} apenas separaria acima e abaixo da média.\n"
           if melhor_k == 2 else
           f", e k={melhor_k} produziria grupos que não se traduzem em intervenções "
           "diferentes.\n")
    )
    L.append("![Escolha de k](../images/16_escolha_k.png)\n")
    L.append("![Perfis em 2D](../images/19_perfis_pca.png)\n")
    L.append("### Unidades com maior risco previsto\n")
    top = municipios.nlargest(15, "risco")[
        ["nome_municipio", "sigla_uf", "rede_descricao", "indicador_previsto",
         "meta_do_ano", "alunos"]
    ]
    L.append(top.to_markdown(index=False, floatfmt=".1f") + "\n")
    L.append("![Municípios em risco](../images/12_municipios_risco.png)\n")

    destino = config.REPORTS_DIR / "modelagem.md"
    destino.write_text("\n".join(L), encoding="utf-8")
    logger.info("Relatório salvo em %s", destino)
    return str(destino)


if __name__ == "__main__":
    gerar()
