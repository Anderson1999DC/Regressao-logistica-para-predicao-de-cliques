# Predição de Clique em Anúncios com Regressão Logística

Este projeto aplica técnicas de **Machine Learning** para prever se um usuário irá **clicar ou não em um anúncio online**, utilizando um modelo de **Regressão Logística**.

O objetivo é demonstrar na prática o fluxo completo de um projeto de **Ciência de Dados**, incluindo análise exploratória, preparação dos dados, modelagem e avaliação do modelo.

---

# Problema

Empresas que investem em **marketing digital** precisam entender quais usuários têm maior probabilidade de interagir com anúncios.

A partir de dados comportamentais e demográficos dos usuários, o objetivo deste projeto é construir um modelo capaz de **prever a probabilidade de clique em anúncios**, permitindo estratégias de segmentação mais eficientes.

---

# Dataset

O conjunto de dados contém informações sobre usuários e seu comportamento de navegação.

Principais variáveis utilizadas:

- **Daily Time Spent on Site** – Tempo médio diário gasto no site (minutos)
- **Age** – Idade do usuário
- **Area Income** – Renda média da região do usuário
- **Daily Internet Usage** – Tempo médio diário de uso da internet
- **Male** – Gênero do usuário (variável binária)
- **Timestamp** – Momento da interação com o anúncio
- **Clicked on Ad** – Variável alvo (0 = não clicou | 1 = clicou)

Algumas variáveis textuais como **City**, **Country** e **Ad Topic Line** foram exploradas mas não utilizadas diretamente na modelagem.

---

# Etapas do Projeto

## 1. Análise Exploratória de Dados (EDA)

Foi realizada uma análise inicial para entender a distribuição das variáveis e identificar padrões entre os usuários que clicam ou não em anúncios.

Foram analisadas distribuições como:

- Idade dos usuários
- Tempo gasto no site
- Uso diário da internet
- Relação entre variáveis comportamentais e cliques

---

## 2. Pré-processamento

As principais etapas de preparação dos dados incluíram:

- Conversão da variável **Timestamp** para formato datetime
- Seleção das variáveis mais relevantes para o modelo
- Separação entre **variáveis independentes (features)** e **variável alvo**
- Divisão do dataset em **treino e teste**

---

## 3. Modelagem

Foi utilizado o algoritmo de **Regressão Logística**, amplamente aplicado em problemas de **classificação binária**.

O modelo foi treinado para identificar padrões que diferenciam usuários que:

- **Clicam em anúncios**
- **Não clicam em anúncios**

---

## 4. Avaliação do Modelo

O desempenho foi avaliado utilizando métricas de classificação:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

O modelo apresentou aproximadamente **90% de acurácia**, indicando boa capacidade de prever o comportamento de clique dos usuários.

---

# Insights de Negócio

A partir dos resultados obtidos, é possível extrair alguns insights relevantes para estratégias de marketing digital:

- Usuários com determinados **padrões de navegação e uso de internet** apresentam maior probabilidade de clicar em anúncios.
- Modelos preditivos podem auxiliar empresas a **segmentar melhor o público-alvo**.
- A utilização de Machine Learning pode **otimizar campanhas publicitárias**, direcionando anúncios para usuários com maior probabilidade de interação.

---

# Limitações do Modelo

Apesar do bom desempenho, algumas limitações devem ser consideradas:

- O modelo utiliza um conjunto relativamente simples de variáveis.
- O desempenho foi avaliado apenas neste conjunto de dados específico.
- Em aplicações reais, seria recomendado testar **outros algoritmos de classificação** e utilizar bases de dados maiores.

---

# Tecnologias Utilizadas

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

---

# Estrutura do Projeto
Regressao-logistica-para-predicao-de-cliques/
│
├── regressao_logistica_predicao_de_click.ipynb
├── advertising.csv
└── README.md

---

# Conclusão

Este projeto demonstra como técnicas de **análise de dados e Machine Learning** podem ser utilizadas para prever o comportamento de usuários em ambientes digitais, contribuindo para decisões mais estratégicas em campanhas de marketing orientadas por dados.

---

## Resultados do Modelo

Principais métricas obtidas no conjunto de teste:

- Accuracy: ~90%
- Precision (classe clique): ~0.95
- Recall (classe clique): ~0.85
- F1-score: ~0.90

Os resultados indicam que o modelo possui boa capacidade de identificar usuários com probabilidade de clicar em anúncios, mantendo equilíbrio entre precisão e sensibilidade.

---

## Como Executar o Projeto

1. Clone este repositório

git clone [https://github.com/seu-usuario/ad-click-prediction-logistic-regression.git](https://github.com/Anderson1999DC/Regressao-logistica-para-predicao-de-cliques)

2. Instale as dependências

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

3. Abra o notebook

