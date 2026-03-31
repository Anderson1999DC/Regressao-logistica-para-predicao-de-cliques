<div align="center">

# Regressão Logística para Predição de Cliques em Anúncios
### EDA · Classificação Binária · Regressão Logística · Marketing Digital

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Concluído-28a745?style=for-the-badge)]()

<br>

> Modelo de classificação binária para prever se um usuário irá clicar em um anúncio online,
> com base em dados comportamentais e demográficos — aplicação direta em estratégias de marketing digital.

</div>

---

## Índice

- [Contexto](#contexto)
- [Objetivos](#objetivos)
- [Pipeline do Projeto](#pipeline-do-projeto)
- [Tecnologias](#tecnologias-utilizadas)
- [Dataset](#dataset)
- [Análise Exploratória](#análise-exploratória)
- [Modelagem e Resultados](#modelagem-e-resultados)
- [Insights de Negócio](#insights-de-negócio)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Autor](#autor)

---

## Contexto

Empresas que investem em **marketing digital** precisam entender quais usuários têm maior probabilidade de interagir com anúncios. Direcionar campanhas sem critério gera desperdício de verba e baixo retorno.

A partir de dados comportamentais e demográficos de 1.000 usuários, o objetivo é construir um modelo capaz de **prever a probabilidade de clique em anúncios**, permitindo segmentação mais inteligente das campanhas.

| Classe | Descrição |
|---|---|
| `0` | Usuário **não clicou** no anúncio |
| `1` | Usuário **clicou** no anúncio |

---

## Objetivos

- Explorar padrões comportamentais que diferenciam usuários que clicam ou não em anúncios
- Identificar as variáveis mais relevantes para a classificação
- Construir e avaliar um modelo de Regressão Logística para classificação binária
- Traduzir os resultados em recomendações para estratégias de marketing digital

---

## Pipeline do Projeto

```mermaid
flowchart TD
    A([advertising.csv\n1.000 usuários · 10 colunas]) --> B[EDA\nHistograma · Jointplots · Pairplot]
    B --> C[Preparação\nConversão Timestamp · Seleção de features]
    C --> D[Regressão Logística\nScikit-learn · 70/30 split]
    D --> E[Avaliação\nConfusion Matrix · Classification Report]
    E --> F([Modelo Final\nAcurácia: 90%])

    B --> B1[/"Dois clusters visíveis\nno pairplot por hue"/]
    E --> E1[/"Precision: 0.95\nRecall: 0.85 · F1: 0.90"/]

    style A fill:#4A90D9,color:#fff,stroke:none
    style F fill:#28a745,color:#fff,stroke:none
    style B fill:#6C757D,color:#fff,stroke:none
    style C fill:#6C757D,color:#fff,stroke:none
    style D fill:#6C757D,color:#fff,stroke:none
    style E fill:#6C757D,color:#fff,stroke:none
```

---

## Tecnologias Utilizadas

| Tecnologia | Uso no Projeto |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Linguagem principal |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Manipulação e análise dos dados |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Operações numéricas |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white) | Histograma de distribuição |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white) | Jointplots e pairplot com hue |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | Modelo de Regressão Logística e métricas |

---

## Dataset

**Fonte:** `advertising.csv` — Dataset fictício de publicidade online
**Uso:** Exclusivamente educacional

| Característica | Detalhe |
|---|---|
| Volume | 1.000 usuários |
| Colunas totais | 10 |
| Features usadas no modelo | 5 |
| Variável alvo | `Clicked on Ad` (0 ou 1) |
| Classes | Balanceadas (~50% / 50%) |

**Variáveis do modelo:**

| Feature | Descrição | Média |
|---|---|---|
| `Daily Time Spent on Site` | Tempo diário no site (min) | 65,0 min |
| `Age` | Idade do usuário | 36,0 anos |
| `Area Income` | Renda média da região (US$) | US$ 55.000 |
| `Daily Internet Usage` | Uso diário de internet (min) | 180 |
| `Male` | Gênero (binária: 0/1) | — |
| `Clicked on Ad` | **Variável alvo** — clicou (1) ou não (0) | — |

---

## Análise Exploratória

### Visão Geral — Separação por Comportamento de Clique

![Pairplot Clicked on Ad](Assets/pairplot_clicked_on_ad.png)

> O pairplot com `hue="Clicked on Ad"` revela **dois clusters bem definidos** — usuários que clicam (vermelho) e os que não clicam (azul) se separam claramente em `Daily Time Spent on Site` e `Daily Internet Usage`. Essa separação visual confirma que o modelo terá base sólida para classificar.

---

### Idade × Tempo no Site (KDE)

![KDE Idade x Tempo no Site](Assets/kde_idade_tempo_site.png)

> Usuários **mais velhos e com menos tempo no site** formam o cluster com maior probabilidade de clicar. Já os usuários mais jovens e com alto tempo no site tendem a **não clicar** — possivelmente mais experientes em ignorar anúncios.

---

### Tempo no Site × Uso Diário de Internet

![Jointplot Tempo no Site x Internet](Assets/jointplot_tempo_site_internet.png)

> Dois grupos distintos: usuários com **alto uso de internet e alto tempo no site** raramente clicam; usuários com **baixo uso de internet e menos tempo no site** clicam mais. Padrão consistente com o perfil de um usuário menos habituado ao ambiente digital.

---

## Modelagem e Resultados

### Métricas de Avaliação

| Métrica | Classe 0 (não clicou) | Classe 1 (clicou) | Média |
|---|---|---|---|
| **Precision** | 0.86 | **0.95** | 0.91 |
| **Recall** | **0.96** | 0.85 | 0.90 |
| **F1-score** | 0.91 | 0.90 | 0.90 |
| **Accuracy** | — | — | **90%** |

### Matriz de Confusão

```
                  Previsto: 0    Previsto: 1
Real: 0 (não clicou)   155            7
Real: 1 (clicou)        25          143
```

> O modelo acertou **155 de 162 não-clicadores** (96% de recall) e **143 de 168 clicadores** (85% de recall). Os 25 falsos negativos representam usuários que clicaram mas não foram identificados pelo modelo — erro aceitável para um primeiro modelo de baseline.

---

## Insights de Negócio

- **Usuários mais velhos com menor uso de internet** têm maior probabilidade de clicar — segmentar campanhas para esse perfil tende a aumentar o CTR
- **Alto tempo no site e alto uso de internet** são indicadores negativos de clique — usuários digitalmente experientes ignoram mais anúncios
- O modelo com **90% de acurácia** permite automatizar a segmentação de público-alvo, reduzindo desperdício de verba em campanhas

### Próximos Passos Sugeridos

- Testar outros algoritmos de classificação (Random Forest, XGBoost) para comparação
- Incluir variáveis de localização (`Country`) via encoding para capturar padrões geográficos
- Aplicar `max_iter` maior no modelo ou normalização de dados para resolver o aviso de convergência
- Implementar uma curva ROC-AUC para avaliar o threshold ideal de classificação

---

## Estrutura do Repositório

```
Regressao-logistica-para-predicao-de-cliques/
│
├── 📁 Assets/                                      # Gráficos gerados na análise
│   ├── pairplot_clicked_on_ad.png
│   ├── kde_idade_tempo_site.png
│   └── jointplot_tempo_site_internet.png
│
├── 📓 regressao_logistica_predicao_de_clique.ipynb  # Notebook completo
├── 📄 advertising.csv                               # Dataset
├── 📄 requirements.txt                              # Dependências do projeto
└── 📄 README.md                                     # Documentação do projeto
```

---

## Autor

<div align="center">

<img src="https://github.com/Anderson1999DC.png" width="100px" style="border-radius:50%"/>

**Anderson Coelho**
*Cientista de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anderson-coelho-42671634a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Anderson1999DC)

</div>

---

<div align="center">

</div>
