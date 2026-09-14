"""Configurações do projeto: caminhos, parâmetros de amostragem e modelagem."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(os.getenv("PROJETO_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
IMAGES_DIR = ROOT_DIR / "images"
MODELS_DIR = DATA_DIR / "models"

# Camadas Gold/Silver produzidas no Tech Challenge da Fase 2
FASE2_DATA = Path(
    os.getenv("FASE2_DATA", ROOT_DIR.parent / "Tech fase 2" / "data")
)

# --- Recorte do problema ----------------------------------------------------
ANO_ALVO = 2024        # ano da avaliação que queremos prever
ANO_BASE = 2023        # ano de onde vêm as features de contexto
SERIE_ALVO = "2"       # 2° ano do Ensino Fundamental
ALFABETIZADO_SIM = "1"

# --- Amostragem -------------------------------------------------------------
# A base tem ~1,85 milhão de alunos válidos. A amostra estratificada mantém a
# proporção do alvo e torna viável validação cruzada e SHAP em tempo razoável.
TAMANHO_AMOSTRA = 150_000
AMOSTRA_SHAP = 5_000
SEED = 42

# --- Modelagem --------------------------------------------------------------
TESTE_PROPORCAO = 0.25
N_FOLDS = 5

# Enriquecimento externo (BigQuery público da Base dos Dados)
ARQUIVO_PROJETO_GCP = ROOT_DIR / ".gcp_project"


def _projeto_gcp() -> str:
    """ID do projeto de faturamento do BigQuery.

    A variável de ambiente tem precedência; sem ela, vale o arquivo local
    `.gcp_project`, que fica fora do versionamento. Assim o ID não precisa ser
    redigitado a cada sessão nem publicado no repositório.
    """
    if os.getenv("GCP_PROJECT_ID"):
        return os.environ["GCP_PROJECT_ID"].strip()
    if ARQUIVO_PROJETO_GCP.exists():
        return ARQUIVO_PROJETO_GCP.read_text(encoding="utf-8").strip()
    return ""


BQ_BILLING_PROJECT = _projeto_gcp()
ARQUIVO_IBGE = DATA_DIR / "externo_ibge.parquet"

# Ano de referência não é data de publicação. A estimativa populacional sai no
# próprio ano de referência; o PIB dos Municípios sai com cerca de dois anos de
# atraso. Como a previsão seria feita no início de ANO_ALVO, o último PIB
# disponível naquela data é o de ANO_BASE - 2.
DEFASAGEM_PUBLICACAO_PIB = 2


def ensure_dirs() -> None:
    for d in (DATA_DIR, REPORTS_DIR, IMAGES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
