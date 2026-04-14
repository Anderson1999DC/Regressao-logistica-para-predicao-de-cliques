# Regressão Logística para Predição de Cliques em Anúncios

### EDA · Classificação · Regressão Logística · ROC-AUC · FastAPI · Docker · Deploy

&nbsp;

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/API-online-28a745?style=for-the-badge)](https://api-predicao-clique.onrender.com)

&nbsp;
> Modelo de classificação para prever a probabilidade de um usuário clicar em um anúncio online,
> com base em características comportamentais e demográficas atingindo ~90% de acurácia.
> Deploy em produção com API REST containerizada.

&nbsp;

**[Acessar interface interativa](https://api-predicao-clique.onrender.com/app)** &nbsp;|&nbsp; **[Documentação da API](https://api-predicao-clique.onrender.com/docs)**

---

## Índice

- [Contexto](#contexto)
- [Objetivos](#objetivos)
- [Pipeline do Projeto](#pipeline-do-projeto)
- [Tecnologias](#tecnologias-utilizadas)
- [Dataset](#dataset)
- [Análise Exploratória](#análise-exploratória)
- [Resultados](#resultados)
- [Insights de Negócio](#insights-de-negócio)
- [API em Produção](#api-em-produção)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Autor](#autor)

---

## Contexto

Projeto de Machine Learning aplicado ao marketing digital, utilizando um dataset fictício de publicidade online. O objetivo é identificar o perfil de usuários com maior probabilidade de interagir com anúncios, permitindo segmentação mais eficiente de campanhas publicitárias. O modelo foi colocado em produção como API REST containerizada.

| Etapa | Descrição |
|---|---|
| **EDA** | Análise de perfil etário, renda regional e padrão de uso da internet |
| **Modelagem** | Regressão Logística para classificação binária |
| **Avaliação** | Acurácia, Precision, Recall, F1-Score e ROC-AUC |
| **Deploy** | API REST com FastAPI + Docker + Render |

---

## Objetivos

- Construir um modelo de classificação para prever cliques em anúncios digitais
- Identificar variáveis comportamentais e demográficas que mais influenciam a decisão de clique
- Avaliar o modelo com métricas completas incluindo ROC-AUC e curva ROC
- Criar uma API REST com FastAPI e containerizar com Docker
- Fazer deploy em produção com link público acessível

---

## Pipeline do Projeto

```mermaid
flowchart TD
    A([Dataset\nPublicidade Digital\n1000 usuários]) --> B[EDA\nIdade · Renda · Tempo Online]
    B --> C[Preparação\nSeleção de features · Timestamp → datetime]
    C --> D[Split Treino/Teste\n67% / 33%]
    D --> E[Regressão Logística\nmax_iter=1000]
    E --> F[Avaliação\nClassification Report · ROC-AUC]
    F --> G[API REST\nFastAPI · Docker]
    G --> H([Deploy\nRender · Link público])

    style A fill:#4A90D9,color:#fff,stroke:none
    style H fill:#28a745,color:#fff,stroke:none
    style B fill:#6C757D,color:#fff,stroke:none
    style C fill:#6C757D,color:#fff,stroke:none
    style D fill:#6C757D,color:#fff,stroke:none
    style E fill:#6C757D,color:#fff,stroke:none
    style F fill:#6C757D,color:#fff,stroke:none
    style G fill:#6C757D,color:#fff,stroke:none
```

---

## Tecnologias Utilizadas

| Tecnologia | Uso no Projeto |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Linguagem principal |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Manipulação e análise dos dados |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Operações numéricas |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | Modelo, métricas e curva ROC |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white) | Curva ROC e visualizações |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white) | Análise exploratória e pairplot |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | API REST para servir o modelo em produção |
| ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | Containerização da aplicação |
| ![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white) | Hospedagem do deploy em produção |

---

## Dataset

**Fonte:** Dataset fictício de publicidade digital criado para fins educacionais
**Uso:** Exclusivamente educacional

| Característica | Detalhe |
|---|---|
| Volume | 1.000 usuários |
| Variável target | `Clicked on Ad` (1 = clicou) |
| Balanceamento | 50% clicou / 50% não clicou |

**Variáveis utilizadas no modelo:**

| Variável | Descrição |
|---|---|
| `Daily Time Spent on Site` | Tempo diário no site (min) |
| `Age` | Idade do usuário |
| `Area Income` | Renda média da região (USD) |
| `Daily Internet Usage` | Uso diário de internet (min) |
| `Male` | Sexo (1 = masculino) |

---

## Análise Exploratória

### Matriz de Confusão

![Matriz de Confusão](Assets/confusion_matrix_clique.png)

> Bom equilíbrio entre verdadeiros positivos e negativos o modelo identifica de forma consistente tanto usuários propensos a clicar quanto aqueles que não clicariam.

### Curva ROC

![Curva ROC](Assets/roc_clique.png)

> ROC-AUC elevado indica forte capacidade do modelo de separar as duas classes usuários que clicam e usuários que não clicam em anúncios.

---

## Resultados

| Métrica | Valor |
|---|---|
| **Acurácia** | **~90%** |
| **Precision (média)** | alto |
| **Recall (média)** | alto |
| **F1-Score (média)** | alto |
| **ROC-AUC** | alto |

> Bom equilíbrio entre Precision e Recall para ambas as classes o modelo identifica de forma consistente tanto usuários propensos a clicar quanto aqueles que não clicariam.

---

## Insights de Negócio

**Perfil de usuário com maior probabilidade de clique:**
- Menor tempo diário no site usuários em busca ativa, não passivos
- Menor uso diário de internet menos expostos a ruído digital
- Faixa etária mais elevada
- Renda regional moderada

**Aplicações práticas:**
- Segmentação de audiência para campanhas de anúncios digitais
- Redução do custo por clique (CPC) ao evitar usuários com baixa propensão
- Score de propensão integrável em plataformas de ad bidding
- Base para sistemas de recomendação de conteúdo patrocinado

**Limitações do modelo:**
- Dataset fictício e simplificado em produção, variáveis como histórico de cliques e categoria do anúncio seriam essenciais
- Avaliado em conjunto de dados único desempenho pode variar em outros contextos

---

## API em Produção

### Interface Interativa

[![Interface do Modelo](Assets/interface.png)](https://api-predicao-clique.onrender.com/app)

> Acesse a interface em: **[api-predicao-clique.onrender.com/app](https://api-predicao-clique.onrender.com/app)**

### Documentação Swagger

[![Swagger UI](Assets/Swagger.png)](https://api-predicao-clique.onrender.com/docs)

> Documentação completa da API em: **[api-predicao-clique.onrender.com/docs](https://api-predicao-clique.onrender.com/docs)**

### Exemplo de Requisição

```bash
curl -X POST https://api-predicao-clique.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "daily_time_spent_on_site": 50.0,
    "age": 35,
    "area_income": 55000.0,
    "daily_internet_usage": 180.0,
    "male": 0
  }'
```

### Resposta

```json
{
  "clicou": 1,
  "resultado": "Alta probabilidade de clique",
  "probabilidade_clique": 0.8234,
  "probabilidade_nao_clique": 0.1766,
  "modelo": "LogisticRegression"
}
```

### Endpoints disponíveis

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` | Status da API |
| `GET` | `/app` | Interface interativa |
| `GET` | `/docs` | Documentação Swagger |
| `POST` | `/predict` | Predição de clique |

---

## Estrutura do Repositório

```
Regressao-logistica-para-predicao-de-cliques/
│
├──  Assets/                                          # Gráficos e imagens
│   ├── confusion_matrix_clique.png
│   ├── roc_clique.png
│   ├── interface.png
│   └── Swagger.png
│
├──  regressao_logistica_predicao_de_clique.ipynb    # Notebook completo
├──  main.py                                          # API FastAPI
├──  index.html                                       # Interface interativa
├──  Dockerfile                                       # Containerização
├──  modelo_predicao_clique.pkl                       # Modelo treinado
├──  colunas_clique.pkl                               # Features esperadas pela API
├──  advertising.csv                                  # Dataset original
├──  requirements.txt                                 # Dependências do projeto
└──  README.md                                        # Documentação do projeto
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
