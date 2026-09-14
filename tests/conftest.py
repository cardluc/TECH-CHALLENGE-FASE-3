"""Mini-base sintética no mesmo formato da base analítica real."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def base_exemplo() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    municipios = [f"350{i:04d}" for i in range(20)]
    df = pd.DataFrame({
        "id_aluno": [f"A{i:05d}" for i in range(n)],
        "id_escola": rng.choice([f"E{i}" for i in range(40)], n),
        "id_municipio": rng.choice(municipios, n),
        "sigla_uf": rng.choice(["SP", "BA", "CE"], n),
        "regiao": rng.choice(["Sudeste", "Nordeste"], n),
        "rede_descricao": rng.choice(["Municipal", "Estadual"], n),
        "caderno": rng.choice(["01", "02", "03"], n),
        "peso_aluno": rng.lognormal(0, 0.2, n).round(3),
        "mun_taxa_ano_anterior": rng.uniform(20, 90, n).round(2),
        "mun_media_ano_anterior": rng.normal(750, 30, n).round(2),
        "uf_taxa_ano_anterior": rng.uniform(30, 80, n).round(2),
        "meta_do_ano": rng.uniform(40, 85, n).round(2),
        "meta_uf_do_ano": rng.uniform(45, 80, n).round(2),
        "alfabetizado": rng.integers(0, 2, n),
    })
    df["dist_meta_mun_uf"] = (df["meta_do_ano"] - df["meta_uf_do_ano"]).round(2)
    df["dist_meta_mun_nacional"] = (df["meta_do_ano"] - 60.0).round(2)
    # nulos reais: municípios sem histórico
    df.loc[df.sample(40, random_state=1).index, "mun_taxa_ano_anterior"] = np.nan
    df.loc[df.sample(25, random_state=2).index, "meta_do_ano"] = np.nan
    return df
