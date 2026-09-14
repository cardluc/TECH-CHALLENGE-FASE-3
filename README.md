# Predição e Inteligência Analítica para Alfabetização no Brasil

> Tech Challenge — Fase 3 | Modelo supervisionado sobre a camada Gold construída na Fase 2

## 1. Contexto do problema

O **Compromisso Nacional Criança Alfabetizada** estabelece que toda criança deve estar alfabetizada até o fim do 2º ano do ensino fundamental, com metas anuais pactuadas até 2030. O INEP mede o avanço pelo **Indicador Criança Alfabetizada (ICA)** — o percentual de estudantes que atingem 743 pontos na escala Saeb.

Saber o resultado depois da avaliação, porém, chega tarde para a gestão. O que um secretário de educação precisa é antecipar: *quais municípios têm maior risco de não alfabetizar suas crianças no próximo ciclo, e o que caracteriza esses lugares*. É esse deslocamento — do diagnóstico para a antecipação — que o projeto endereça.

## 2. Objetivo analítico

Construir um modelo supervisionado que preveja se um aluno será considerado **alfabetizado ou não alfabetizado**, a partir de variáveis **educacionais** (histórico de desempenho do município e do estado, meta pactuada), **territoriais** (UF, região, rede administrativa) e **socioeconômicas** (população e PIB municipal).

A restrição que organiza o projeto: só entra variável disponível **antes** da avaliação. O modelo estima probabilidade, não rótulo — o corte de decisão é calibrado depois, em função do custo de cada tipo de erro (seção 6).

A restrição é deliberada. Prever alfabetização usando a proficiência do próprio aluno seria trivial e inútil — o alvo é derivado dela. O modelo só tem valor se antecipar risco com informação que a gestão já possui no início do ano letivo.

## 3. Descrição da base utilizada

Origem: camada **Gold** do Tech Challenge da Fase 2 (`gold_indicador_municipio`, `gold_indicador_uf`, `gold_evolucao_brasil`), alimentada pelo dataset `br_inep_avaliacao_alfabetizacao` (INEP, via Base dos Dados).

Três leituras recorrem à Silver, por motivos que a Gold não resolve:

| O que | Por quê |
|---|---|
| `fato_aluno` | o alvo é por aluno; a Gold agrega |
| microdado de 2023 | a Gold não materializa quartis, e a dispersão é a 4ª variável mais influente |
| `oficial_municipio` | a Gold é construída *a partir* do microdado, então perde os municípios sem aluno avaliado em 2023; a taxa oficial cobre esses casos e eleva a cobertura do indicador de 77% para 93% |

| Item | Valor |
|---|---|
| Unidade de análise | um aluno avaliado em 2024 (2º ano do EF) |
| Registros | **1.851.852 alunos válidos**, 25 colunas |
| Alvo | `alfabetizado` (1/0) — 59,8% positivos |
| Contexto | desempenho de 2023, por município e UF |
| Enriquecimento externo | população de 2023 e PIB municipal de 2021 (IBGE, via BigQuery) |

**Ler da Gold exige mais cuidado com vazamento, não menos.** A partição de 2024 já traz `indicador_alfabetizacao`, `taxa_oficial`, `gap_meta` e `atingiu_meta` — ou seja, o resultado calculado. Por isso todo contexto é filtrado em 2023, e da partição do ano-alvo só se extrai a **meta**, que é pactuada com antecedência. As quatro colunas acima estão na lista de proibidas do teste de vazamento.

**Variáveis preditoras** (todas anteriores ao resultado ou pactuadas com antecedência):

**Educacionais** — o histórico de desempenho e o compromisso pactuado:

- `mun_taxa_ano_anterior` — **Indicador Criança Alfabetizada** do município em 2023
- `mun_media_ano_anterior` — proficiência média do município em 2023
- `mun_desvio_ano_anterior`, `mun_p25_ano_anterior`, `mun_p75_ano_anterior` — dispersão do desempenho municipal
- `mun_alunos_ano_anterior` — porte da avaliação no município
- `uf_taxa_ano_anterior`, `uf_media_ano_anterior` — desempenho estadual
- `meta_do_ano`, `dist_meta_partida` — **meta municipal** pactuada e distância do ponto de partida
- `meta_uf_do_ano`, `dist_meta_mun_uf` — **meta estadual** e o quanto a meta local a supera
- `dist_meta_mun_nacional` — distância da meta local à **meta nacional** da rede Pública (60,0% em 2024)

**Territoriais** — onde e sob qual administração o aluno estuda:

- `sigla_uf`, `regiao` — unidade federativa e macrorregião
- `rede_descricao` — rede administrativa (municipal, estadual, federal, privada)

**Socioeconômicas** — o contexto material do município:

- `populacao` — estimativa do IBGE para 2023
- `pib`, `pib_per_capita` — PIB dos Municípios de 2021, o último publicado antes do ciclo (ver 6.4)

**Técnica** — controle do instrumento de medida:

- `caderno` — versão da prova aplicada, para absorver diferença de dificuldade entre cadernos

Só entra no modelo o que está nessa lista. O pipeline usa **lista branca**, não lista negra: qualquer coluna nova é descartada com aviso no log, e um teste automatizado injeta variáveis proibidas para confirmar que são barradas.

**Duas armadilhas da fonte, identificadas e tratadas:**

1. **A distribuição por nível de desempenho só existe a partir de 2024.** Em 2023 essas colunas são inteiramente nulas. A dispersão do desempenho municipal foi reconstruída a partir do microdado (desvio-padrão e quartis da proficiência).
2. **Não há nenhuma variável de escola, e isso é deliberado.** O `id_escola` é máscara regerada a cada ano: apenas 3,5% dos códigos presentes nos dois anos pertencem ao mesmo município, e a taxa por escola correlaciona 0,25 entre 2023 e 2024, contra 0,64 da taxa municipal — usar o histórico seria alimentar ruído. E o porte medido no próprio ano também não serve: contar alunos avaliados em 2024 só é possível *depois* da prova. Não é vazamento, já que não contém a resposta; é **indisponibilidade na data da previsão**. Uma variável de escola exigiria o Censo Escolar, que traz infraestrutura e corpo docente conhecidos antes do ciclo.

## 4. Etapas de modelagem

```
base analítica → EDA → pipeline sklearn → validação cruzada agrupada
              → otimização → avaliação no teste → interpretabilidade → aplicação
```

O pré-processamento está **dentro** do `Pipeline` do scikit-learn (`ColumnTransformer`), o que garante que mediana de imputação, média/desvio da padronização e categorias do encoding sejam estimadas **apenas no treino**, a cada dobra:

- **Numéricas**: `SimpleImputer(median, add_indicator=True)` + `StandardScaler`
- **Categóricas**: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`

O `add_indicator` é uma decisão analítica, não técnica: o valor faltante aqui significa *"município sem avaliação no ano anterior"* — a ausência é informação.

### Tratamento de data leakage

Três camadas de proteção:

1. **Na construção da base** — nenhuma variável do próprio ano entra como feature. Proficiência, nível de desempenho e taxa de 2024 ficam de fora; o alvo deriva deles.
2. **Na separação treino/teste** — o split é feito **por município** (`GroupShuffleSplit`). Como as variáveis descrevem o contexto municipal, deixar o mesmo município nos dois lados permitiria ao modelo memorizar o contexto em vez de generalizar. O teste mede desempenho em municípios **nunca vistos**.
3. **Na validação cruzada** — `GroupKFold`, pelo mesmo motivo, em todas as dobras e na busca de hiperparâmetros.

Há testes automatizados que falham se qualquer uma dessas regras for violada.

## 5. Escolha do algoritmo

Uma representante de cada família vista no curso, em validação cruzada agrupada por município (5 dobras):

| família | modelo | ROC AUC (CV) | desvio | gap treino |
|---|---|---|---|---|
| ensemble — boosting | HistGradientBoosting | **0,6543** | ±0,0102 | 0,0259 |
| ensemble — bagging | Random Forest | 0,6537 | ±0,0100 | 0,0310 |
| linear probabilística | Regressão Logística | 0,6482 | ±0,0175 | 0,0101 |
| margem máxima | SVM linear | 0,6463 | ±0,0164 | 0,0122 |
| árvore | Árvore de Decisão | 0,6427 | ±0,0083 | 0,0262 |
| bayesiana | Naive Bayes | 0,6276 | ±0,0153 | 0,0162 |
| — | Baseline (classe majoritária) | 0,5000 | ±0,0000 | 0,0000 |

**HistGradientBoosting e Random Forest estão empatados** — 0,0006 de diferença contra desvio de ±0,010 entre dobras não sustenta afirmar superioridade. Adotei o HistGradientBoosting por dois motivos práticos: aceita busca de hiperparâmetros mais fina e treina mais rápido. Se o critério fosse exclusivamente a métrica, a escolha seria indiferente.

O empate entre seis famílias é o primeiro indício de que o limite é a **informação disponível**, não a capacidade do modelo. A seção 6.2 confirma com a curva de aprendizado: mais dados também não resolvem.

Hiperparâmetros por busca aleatória (10 combinações) com validação cruzada agrupada.

## 6. Métricas de avaliação

Conjunto de teste (1.346 municípios que o modelo nunca viu), ponderado pelo peso amostral:

| métrica | valor |
|---|---|
| ROC AUC | **0,6695** |
| PR AUC | 0,7399 |
| Acurácia | 0,6108 |
| Brier score | 0,2209 |

Limiar de alerta: **0,62**, escolhido na validação agrupada do treino (alvo: recall ≥ 0,70 para a classe não alfabetizada). O conjunto de teste não participou dessa escolha.

| classe | precisão | recall | F1 |
|---|---|---|---|
| alfabetizado | 0,7227 | 0,5585 | 0,6301 |
| **não alfabetizado** | **0,5160** | **0,6872** | **0,5894** |

**A classe que interessa é a negativa.** O modelo tem `alfabetizado` como classe positiva, mas quem a política pública precisa encontrar é a criança **não** alfabetizada. No limiar padrão de 0,5 o recall dessa classe fica em torno de 0,40 — a maioria dos casos de risco não seria sinalizada. Deslocando o corte para 0,62, o recall sobe para **0,687** ao custo de precisão de 0,516: de cada 10 crianças não alfabetizadas, ~7 são sinalizadas; de cada 10 sinalizadas, ~5 são alarme falso.

Essa troca é deliberada e é a razão de o limiar não ser 0,5: **deixar de sinalizar uma criança em risco custa mais do que investigar um falso alerta.** A escolha foi feita na validação cruzada agrupada do treino, com o teste reservado.

**O limiar se sustenta fora do conjunto onde foi escolhido.** O corte foi definido mirando recall de 0,70 na validação cruzada do treino e entrega 0,687 em municípios que o modelo nunca viu — pouco mais de 1 ponto percentual de diferença. Isso diz respeito ao **comportamento do corte**, não à calibração das probabilidades: essa é medida pelo Brier score e pela curva de calibração, reportados à parte. Em produção o limiar deveria ser redefinido a cada ciclo, já que o recall depende da composição municipal.

O Brier score, com a curva de calibração, mostra que as probabilidades são utilizáveis como medida contínua de risco — não apenas como rótulo sim/não.

### 6.1 O produto final validado: a taxa municipal prevista

O ROC AUC por aluno não prova que o ranking municipal sirva para decidir. A verificação direta é o erro da taxa prevista, em pontos percentuais, contra a regra ingênua *"repetir o indicador do ano anterior"*:

| | MAE | RMSE |
|---|---|---|
| Modelo | **8,33 p.p.** | 11,00 p.p. |

| Regra do ano anterior | 11,04 p.p. | — |

O modelo erra **25% menos** que a regra ingênua (4.092 unidades comparadas). É este número, e não o AUC, que sustenta o uso do ranking para priorização.

### 6.2 Duas hipóteses testadas que não se confirmaram

**Escalonamento robusto.** A hipótese era que a mediana e o IQR tratariam melhor a assimetria de população e PIB, recuperando o desempenho da regressão logística. Não recuperou: **0,6482 com z-score contra 0,6472 com escalonamento robusto**. As árvores ficaram idênticas, como esperado — são invariantes a escala. Hipótese descartada com evidência.

**Mais dados.** A curva de aprendizado vai de 2.666 a 26.666 exemplos. O AUC de validação sobe no trecho todo, mas a subida se concentra no início: os últimos 22% de dados (21.866 → 26.666) rendem **0,0017**. A curva achatou dentro do intervalo avaliado.

Isso **não** demonstra que mais dados nunca ajudariam — testar essa afirmação exigiria estender a curva bem além de 26 mil exemplos. O que os dados sustentam é mais modesto e ainda assim útil: *no intervalo medido, o retorno marginal do volume já é desprezível*. O gap treino–validação, entre 0,010 e 0,031 nos seis algoritmos, mostra que também não há sobreajuste a corrigir. Os números exatos de cada rodada estão em `reports/curva_aprendizado.csv` e `comparacao_modelos.csv`.

A curva de validação reforça: o desempenho máximo aparece com `max_leaf_nodes = 4`. Um modelo que precisa de apenas quatro folhas está descrevendo um sinal simples, não uma estrutura complexa mal capturada.

### 6.3 Cobertura das variáveis de contexto

O ICA municipal do ano anterior falta em 7,2% dos registros; as variáveis reconstruídas do microdado (desvio, p25, p75 e contagem de alunos) faltam em 23,2%. A diferença é estrutural: elas exigem ter havido aluno avaliado naquele município em 2023, condição mais restritiva que a publicação da taxa oficial agregada — que é justamente o que preenche os 16 pontos de diferença entre as duas coberturas (ver `contexto_municipio`).

Esses ausentes não são descartados. O `add_indicator` do imputador cria, para cada coluna com nulo, uma companheira marcando a ausência — o modelo recebe o valor imputado *e* o aviso de que foi imputado.

### 6.4 Enriquecimento externo: a ablação e o que ela mostrou

População e PIB municipal do IBGE entraram na base como hipótese — *o contexto socioeconômico melhora a previsão?* A ablação é executada **dentro do pipeline**, a cada rodada: os mesmos sete estimadores são comparados com e sem as três colunas, na mesma amostra, mesmo split e mesmas dobras. O resultado fica em `reports/ablacao_ibge.csv` e na seção correspondente de `reports/modelagem.md`, para quem quiser conferir sem reexecutar.

Na execução de referência, as diferenças vão de **−0,0058** (naive bayes, que piora com as variáveis) a **+0,0005** (regressão logística). Todas ficam abaixo do desvio entre dobras de qualquer um dos algoritmos, que gira em torno de ±0,010 — ou seja, são indistinguíveis de ruído.

As variáveis foram **mantidas** porque a integração socioeconômica é requisito do projeto, e porque o modelo de fato as usa (`populacao` aparece na 8ª posição do SHAP, `pib_per_capita` na 12ª). O que a ablação mostra é que o sinal que elas carregam já está disponível nas variáveis de desempenho territorial.

**Ano de referência não é data de publicação.** Essa distinção quase passou batida e vale explicitar: a estimativa populacional do IBGE sai no próprio ano de referência, mas o PIB dos Municípios é publicado com cerca de dois anos de atraso. Travar as duas fontes em 2023 teria colocado na base um PIB que só passou a existir depois da prova que o modelo prevê — a mesma categoria de problema da variável de porte de escola que foi removida: **indisponibilidade na data da previsão**, não vazamento do alvo.

Por isso cada fonte tem seu próprio teto (`config.DEFASAGEM_PUBLICACAO_PIB`): população até `ANO_BASE`, PIB até `ANO_BASE - 2`. Os anos efetivos são gravados no arquivo e reconferidos na leitura — um parquet fora do teto faz a execução parar, em vez de seguir silenciosamente.

## 7. Interpretação dos resultados

Duas técnicas independentes — **importância por permutação** e **SHAP** — concordam no ranking:

| variável | SHAP | permutação | leitura |
|---|---|---|---|
| `mun_taxa_ano_anterior` | **0,182** (1º) | **0,0206** (1º) | o ICA do município no ano anterior |
| `uf_media_ano_anterior` | 0,142 (2º) | 0,0099 (3º) | proficiência média do estado |
| `mun_media_ano_anterior` | 0,121 (3º) | 0,0103 (2º) | proficiência média do município |
| `mun_p75_ano_anterior` | 0,112 (4º) | 0,0037 (5º) | cauda superior da distribuição municipal |

**Os métodos concordam no primeiro lugar e trocam o segundo com o terceiro.** A diferença na permutação é de 0,0004 — bem dentro do ruído de um método que embaralha colunas aleatoriamente. Vale registrar assim, e não como concordância perfeita: `uf_media` e `mun_media` medem coisas correlacionadas, e qual delas "vence" depende de detalhes do procedimento.

O que os dois métodos afirmam com segurança é o mesmo: **o ICA municipal do ano anterior domina, e logo atrás vêm outras três medidas de contexto territorial**. As quatro somam 0,557 de SHAP, e nenhuma descreve a criança.

A leitura de negócio é direta e desconfortável: **o principal preditor do desempenho de uma criança é o desempenho do lugar onde ela estuda**. Não é uma revelação sobre a criança, é uma medida da desigualdade territorial da política educacional.

## 8. Insights encontrados

1. **Desigualdade regional acentuada** — a alfabetização vai de 64,7% no Centro-Oeste a 50,8% no Norte. Entre UFs, a distância entre a melhor e a pior chega a **49,3 pontos percentuais**.
2. **47,8% das unidades em risco** — 1.746 de 3.656 unidades **município × rede que têm meta pactuada** ficam abaixo dela. Outras 533 unidades não têm meta e são classificadas como *não avaliáveis*, não como "sem risco" — a agregação é por município × rede porque a meta é pactuada por rede.
3. **O risco acompanha a geografia** — Norte (48,5%) e Nordeste (43,8%) concentram o risco; Sudeste (31,7%) e Centro-Oeste (32,3%) estão na ponta oposta. Este é o achado **mais estável** entre execuções: a ordem das regiões e a distância entre os extremos praticamente não se movem.
4. **Existe um grupo que a meta deixou para trás** — o perfil "Atenção" reúne 1.571 unidades cuja previsão (58,9%) fica próxima do próprio histórico (59,7%) mas quatro pontos abaixo da meta (63,0%). Não são municípios em queda: são municípios estáveis com meta acima do ritmo que vêm praticando. A formulação correta é *a previsão não alcança a meta*, não *estão piorando*.
5. **A dispersão importa tanto quanto a média** — `p25` e `p75` municipais aparecem entre as variáveis mais influentes: municípios com desempenho heterogêneo escondem escolas em risco atrás de uma média aceitável.
6. **O estado pesa quase tanto quanto o município** — `uf_media_ano_anterior` (SHAP 0,142) fica a um passo de `mun_taxa_ano_anterior` (0,182) e no mesmo patamar da própria média municipal (0,121). Duas crianças em municípios equivalentes de estados diferentes recebem previsões diferentes: a unidade federativa carrega sinal que o município não explica.
7. **Nem algoritmo nem volume de dados aparecem como gargalo** — seis famílias entre 0,628 e 0,654, gap treino–validação abaixo de 0,032, e retorno marginal de 0,0017 no último passo da curva de aprendizado. Três evidências convergindo para a mesma leitura: o que falta é informação sobre a criança. Convergência não é prova — mas é o que se pode afirmar sem dados que não existem.

### Perfis de município (k-means)

| perfil | municípios | indicador previsto | taxa ano anterior | meta |
|---|---|---|---|---|
| Crítico | 822 | 39,6% | 34,6% | 41,3% |
| Atenção | 1.571 | 58,9% | 59,7% | 63,0% |
| Intermediário | 573 | 64,6% | 49,0% | 53,9% |
| Consolidado | 1.223 | 78,9% | 79,7% | 76,5% |

**Os perfis são o resultado menos estável do projeto.** Entre execuções que diferem apenas por qual ano do PIB entra na base, o tamanho do grupo "Atenção" oscilou entre 297 e 1.571 unidades, e o caráter dele mudou junto. A silhueta baixa (0,354) já anuncia isso: a estrutura de grupos é fraca, e pequenas mudanças nas variáveis reorganizam as fronteiras.

A leitura defensável é a **tipologia** — existem municípios consolidados, críticos, intermediários e um grupo cuja previsão não acompanha a meta — não a contagem exata de cada grupo. Um número de perfil não deve ser usado para decidir sobre um município específico sem olhar os valores dele.

**A silhueta máxima está em k=2 (0,401), não em k=4 (0,354).** A escolha de quatro perfis é uma decisão de negócio, não estatística, e precisa ser defendida como tal: k=2 separaria apenas "acima" e "abaixo" da média, o que não distingue ações. Os quatro perfis correspondem a intervenções diferentes — *Crítico* pede reforço estrutural; *Atenção* pede investigação urgente (previsão 11 pontos abaixo do próprio histórico); *Intermediário* pede manutenção do ritmo; *Consolidado* pede apenas monitoramento. Se o critério fosse exclusivamente a coesão dos grupos, k=2 venceria.

## 9. Limitações do projeto

Sendo direto sobre o que este modelo **não** é:

1. **O ROC AUC de 0,67 é modesto — e isso é esperado.** Sem nenhuma variável individual da criança (frequência, histórico escolar, condição socioeconômica familiar), o modelo estima o risco do **ambiente**, não o desempenho pessoal. O teto de previsibilidade é o contexto, e o empate entre algoritmos confirma isso.
2. **Predição individual não deve ser usada para decidir sobre uma criança específica.** O uso legítimo é priorizar territórios e escolas, nunca rotular alunos.
3. **Apenas dois anos de dados.** Sem série histórica longa, não é possível separar tendência de flutuação nem validar o modelo em múltiplos ciclos.
4. **Associação, não causalidade — e o risco é de viés de seleção.** As variáveis mais influentes indicam *onde* o risco se concentra, não *o que o causa*. Municípios com histórico melhor diferem dos demais **antes** de qualquer intervenção (renda, gestão, infraestrutura são variáveis confundidoras), então a comparação entre grupos não isola efeito. O ranking responde "quem está em risco?", não "o que funciona?" — esta segunda pergunta exigiria contrafactual: experimento aleatorizado, pareamento por *propensity score* ou desenho quase-experimental.
5. **Dados de corte transversal, não série temporal.** Há um único ciclo-alvo com covariável defasada, não uma sequência {Yₜ}. Por isso não se aplicam decomposição sazonal, testes de estacionariedade nem *backtest* walk-forward: a ordem temporal é respeitada por construção (features de 2023, alvo em 2024), e o split é agrupado por município, não por data.
6. **O nível da escola foi perdido** por limitação da fonte (máscara regerada anualmente) — era a unidade mais próxima do aluno.
7. **A rede privada tem 24 alunos** na amostra: qualquer leitura sobre ela é estatisticamente vazia.
8. **Modelo treinado em amostra** de 150 mil alunos para viabilizar validação cruzada e SHAP; o ranking municipal usa 600 mil.

### 9.1 Decisões metodológicas e o que foi testado

Cada escolha abaixo foi verificada empiricamente, não assumida. Os artefatos ficam em `reports/`.

| Decisão | Alternativas testadas | Evidência |
|---|---|---|
| Algoritmo | seis famílias, contra um baseline de classe majoritária (0,5000): gradient boosting 0,6543 · random forest 0,6537 · regressão logística 0,6482 · SVM linear 0,6463 · árvore 0,6427 · naive bayes 0,6276 | `comparacao_modelos.csv` |
| Enriquecimento externo | ablação com e sem população e PIB, executada dentro do pipeline nas mesmas dobras | `ablacao_ibge.csv` |
| Diagnóstico viés-variância | erro de treino e de validação reportados lado a lado; curva de aprendizado e curva de validação | `curva_aprendizado.csv`, `curva_validacao.csv`, imagens 17 e 18 |
| Escalonamento | z-score contra escalonamento robusto (mediana e IQR), justificado pela assimetria de população e PIB | `comparacao_escalador.csv` |
| Codificação categórica | one-hot com agrupamento de categorias raras; `id_municipio` **não** recebe *target encoding* — reintroduziria a memorização que o split agrupado existe para impedir | `pipeline.py` |
| Número de perfis | k de 2 a 8, avaliado por cotovelo, silhueta e Davies-Bouldin, em três famílias de clusterização (k-means, hierárquico Ward e mistura gaussiana) | `escolha_k.csv`, imagem 16 |
| Validação do produto | erro da taxa municipal em p.p. contra a regra "repetir o ano anterior" | `aplicacao_estrategica.json` |
| Limiar de decisão | calibrado na validação do treino por recall **ponderado** da classe não alfabetizada, com o teste reservado | `metricas.json` |
| Peso amostral | aplicado em todo ajuste — comparação de algoritmos, curvas, busca de hiperparâmetros, modelo final, previsões *out-of-fold* e importância por permutação. O *scorer* da validação cruzada permanece sem peso: ali o objetivo é ranquear algoritmos ajustados em condições idênticas, não estimar a taxa populacional | `treinar.pesos()` |
| Interpretabilidade | dois métodos independentes (permutação e SHAP), com *beeswarm* e *dependence plot* além do ranking | imagens 11, 13, 14 e 15 |

## 10. Aplicação estratégica

### 10.1 As cinco perguntas de negócio

**Quais fatores mais impactam a alfabetização?**
O ICA do município no ano anterior domina (SHAP 0,182), seguido pela proficiência média do estado (0,142) e pela média do próprio município (0,121). Permutação e SHAP concordam no primeiro lugar e trocam entre si o segundo e o terceiro — ver seção 7. A leitura é desconfortável: o principal preditor do resultado de uma criança é o desempenho do lugar onde ela estuda, não algo sobre ela.

**Quais municípios apresentam maior risco educacional?**
O ranking completo está em `reports/risco_municipal.csv`, ordenável por risco previsto, com nome, UF, rede, indicador previsto, meta e número de alunos. As 15 primeiras posições aparecem em `reports/modelagem.md` e na imagem 12. As previsões são *out-of-fold*: cada unidade é prevista por um modelo que não viu aquele município no treino.

**Quais regiões possuem padrões semelhantes?**
Dois recortes. Por macrorregião, o risco médio vai de 48,5% no Norte a 31,7% no Sudeste — e essa ordem é o resultado mais estável do projeto entre execuções. Por comportamento, o k-means agrupa as unidades em quatro perfis — Crítico, Atenção, Intermediário e Consolidado — que não coincidem com a geografia: um município do Sudeste pode estar no perfil Crítico. O agrupamento foi validado por silhueta, Davies-Bouldin e comparação entre k-means, hierárquico Ward e mistura gaussiana; a composição dos grupos, porém, é instável (ver seção 8).

**Como prever municípios que podem não atingir metas futuras?**
Comparando o indicador previsto com a meta pactuada, por município × rede. Resultado: 1.746 de 3.656 unidades com meta (47,8%) ficam abaixo dela. A validação desse produto está na seção 6.1 — o erro da taxa prevista é de 8,33 p.p. contra 11,04 p.p. da regra de repetir o ano anterior.

**Quais variáveis possuem maior influência nos modelos?**
Ver seção 7 e as imagens 11, 13, 14 e 15 (permutação, SHAP em barras, *beeswarm* e *dependence plot*). Dois achados além do ranking: a proficiência média do estado fica no mesmo patamar da do próprio município, e a dispersão (p75, p25) entra entre as mais influentes — municípios heterogêneos escondem escolas em risco atrás de uma média aceitável.

### 10.2 Como o modelo rodaria num ciclo futuro

A validação aqui prova generalização para **municípios** que o modelo nunca viu — não para **anos** futuros. São coisas diferentes, e vale ser explícito sobre o que seria preciso para usar isto em 2026.

**Quando a previsão seria feita.** No início do ano letivo, antes da aplicação da prova. Essa data é o que define quais variáveis podem entrar: é por isso que o histórico é sempre do ano anterior, que a meta é a pactuada com antecedência, e que o PIB recua dois anos (o IBGE ainda não publicou o do ano-base).

**O que estaria disponível nessa data.** Tudo que a base usa hoje, com uma defasagem a mais:

| variável | origem no ciclo seguinte |
|---|---|
| ICA e proficiência do município e da UF | resultado do ciclo anterior, já publicado |
| dispersão municipal (desvio, p25, p75) | microdado do ciclo anterior |
| metas municipal, estadual e nacional | pactuadas antes do ciclo |
| população e PIB | IBGE, com a defasagem de publicação de cada fonte |

Nenhuma exige dado novo: o pipeline da Fase 2 reprocessado com o ciclo mais recente entrega todas.

**O que precisaria ser refeito, não reaproveitado.** O modelo salvo em `data/models/` **não** deve ser aplicado direto no ciclo seguinte. Três coisas mudam de ano para ano:

1. **O limiar.** Ele foi calibrado para recall de 0,70 na validação deste ciclo. Como o recall depende da composição municipal, ele precisa ser recalibrado com os dados do ciclo anterior antes de cada uso.
2. **A distribuição das variáveis.** Se a composição das escolas avaliadas mudar, o modelo degrada por *data drift*; se a régua mudar — uma revisão do corte de 743 pontos, por exemplo —, degrada por *concept drift*. Os dois são detectáveis comparando as distribuições de entrada entre ciclos.
3. **A própria relação.** Dois anos de dados não permitem afirmar que o padrão de 2023→2024 vale para 2025→2026.

**Como isso seria validado.** Com um terceiro ciclo disponível, a verificação honesta é treinar em 2023→2024 e testar em 2024→2025, medindo o mesmo MAE da taxa municipal contra a mesma regra ingênua. Só esse teste responde se o modelo generaliza no tempo — e ele não pode ser feito com os dados que existem hoje.

### 10.3 Uso prático para políticas públicas

- **Priorização orçamentária** — a lista dos municípios com maior risco previsto, ordenada, é diretamente acionável para direcionar formação de professores, material e apoio técnico antes do ciclo letivo.
- **Alerta precoce** — o perfil "Atenção" reúne 1.571 unidades estáveis no próprio histórico, mas com previsão abaixo da meta pactuada. Não são municípios em queda: são municípios cuja meta foi fixada acima do ritmo que eles vêm praticando. É onde a conversa sobre viabilidade da meta precisa acontecer antes do ciclo, não depois.
- **Definição realista de metas** — comparar o indicador previsto com a meta pactuada mostra onde a meta está em risco no ritmo atual. "Em risco" não é "impossível": a previsão descreve a trajetória sem intervenção, e o propósito de sinalizar é justamente permitir que haja intervenção. O que o dado sustenta é a conversa de repactuação, não a sentença.
- **Foco na dispersão** — municípios com `p25` muito abaixo da média têm escolas em situação crítica escondidas por uma média aceitável: são candidatos a intervenção cirúrgica, não a programas genéricos.

## 11. Possíveis evoluções futuras

1. **Variáveis individuais e escolares** — Censo Escolar (infraestrutura, formação docente, distorção idade-série) elevaria o teto de previsibilidade que hoje limita o modelo.
2. **Identificador estável de escola** — negociar com o INEP o acesso a um código rastreável entre anos habilitaria o nível mais relevante de intervenção.
3. **Série histórica mais longa** — com 4+ ciclos, modelar trajetória (crescimento, estagnação, queda) em vez de foto anual.
4. **Inferência causal** — diferenças-em-diferenças ou desenho de regressão descontínua sobre municípios que receberam programas específicos, para sair da associação.
5. **Recuar o ano do IBGE até o último efetivamente publicado** — a ablação (6.4) mostrou que as variáveis socioeconômicas não alteram o resultado, mas a defasagem entre ano de referência e data de publicação continua sendo uma imprecisão a corrigir num uso real.
6. **Monitoramento de deriva** — o modelo degrada se a composição das escolas mudar (*data drift*) ou se a própria régua mudar, por exemplo uma revisão do corte de 743 pontos (*concept drift*). Acompanhar a distribuição das variáveis e recalibrar a cada ciclo, aproveitando o pipeline da Fase 2.
7. **Publicação como serviço** — expor as previsões municipais em painel ou API para as secretarias estaduais.

## 12. Estrutura do repositório

```
├── data/                      # base analítica e modelo (não versionados)
├── notebooks/resumo.ipynb     # leitura guiada dos resultados
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── base_analitica.py      # monta a base por aluno, sem vazamento
│   │   ├── enriquecimento_ibge.py # população e PIB via BigQuery
│   │   ├── pipeline.py            # ColumnTransformer (imputação, escala, encoding)
│   │   └── eda.py                 # análise exploratória
│   ├── modeling/treinar.py        # split agrupado, comparação, otimização
│   ├── evaluation/
│   │   ├── avaliar.py             # métricas, permutação, SHAP
│   │   └── aplicacao_estrategica.py  # risco municipal e perfis
│   ├── visualization/graficos.py
│   └── relatorio.py
├── reports/                   # eda.md, modelagem.md, métricas e rankings
├── images/                    # 19 gráficos gerados
├── tests/                     # 13 testes (vazamento, split, pipeline, base)
├── main.py                    # pipeline reproduzível de ponta a ponta
├── requirements.txt
└── README.md
```

## 13. Como executar

**Pré-requisito:** a camada Silver/Gold da Fase 2 disponível. O caminho padrão é `../Tech fase 2/data`; para outro local, defina `FASE2_DATA`.

```bash
python -m pip install -r requirements.txt

# enriquecimento externo (uma vez; requer BigQuery Sandbox)
# o ID do projeto de faturamento vem do arquivo .gcp_project (não versionado)
# ou da variável de ambiente GCP_PROJECT_ID, que tem precedência
"<ID do seu projeto>" | Set-Content .gcp_project -NoNewline
python -m src.preprocessing.enriquecimento_ibge

# pipeline completo
python main.py

# ou uma etapa por vez
python main.py --etapa eda
python main.py --etapa treino

# testes
python -m pytest tests -v
```

Reprodutibilidade garantida por semente fixa (`config.SEED = 42`) em amostragem, split, validação cruzada e modelos.

## 14. Fluxo de trabalho com Git

Commits incrementais por etapa analítica — base, EDA, pipeline, comparação de algoritmos, diagnóstico viés-variância, avaliação, aplicação estratégica e testes. Branches `feat/` e `fix/`, integração via Pull Request. As decisões que sustentam cada etapa ficam registradas nos artefatos de `reports/`, não só no texto.

---

*Projeto acadêmico — FIAP/POSTECH, Tech Challenge Fase 3. Dados públicos do INEP e IBGE.*
