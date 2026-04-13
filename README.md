# Análise de Risco de Crédito

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?logo=scikitlearn)

Análise de risco de crédito com 3 modelos de Machine Learning. Compara **Random Forest**, **Gradient Boosting** e **Logistic Regression** em 1.000 registros do dataset German Credit.

## Resultados

| Modelo | Acurácia | AUC-ROC |
|---|---|---|
| **Random Forest** | 78.0% | **0.784** |
| Gradient Boosting | 75.5% | 0.779 |
| Logistic Regression | 76.5% | 0.742 |

**Cross-validation (Random Forest):** 73.5% ± 2.0%

### Feature Importance

```
Credit amount             0.2462  ████████████████████
Age                       0.1833  ███████████████
Duration                  0.1489  ████████████
Checking account          0.1329  ███████████
Purpose                   0.0904  ███████
Saving accounts           0.0708  ██████
Job                       0.0520  ████
Housing                   0.0455  ████
Sex                       0.0301  ██
```

## Stack

- **Python** — linguagem principal
- **Scikit-Learn** — Random Forest, Gradient Boosting, Logistic Regression, cross-validation
- **Pandas / NumPy** — manipulação de dados
- **Matplotlib / Seaborn** — visualizações (ROC curve, confusion matrix, feature importance)

## Como Executar

```bash
# 1. Clonar
git clone https://github.com/murillosezerino/analise-risco-credito.git
cd analise-risco-credito

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar análise (treina 3 modelos e gera gráficos)
python src/analise_risco_credito.py

# 4. Rodar testes
pytest tests/ -v
```

### Saídas geradas

- **Matriz de Confusão** — heatmap do melhor modelo (Random Forest)
- **Curva ROC** — comparativo dos 3 modelos
- **Feature Importance** — barplot das variáveis mais preditivas
- **Comparativo** — acurácia e AUC lado a lado

## Estrutura do Projeto

```
├── src/
│   └── analise_risco_credito.py  # Script completo: load → preprocess → train → evaluate
├── dados/
│   └── dados_credito.csv         # German Credit dataset (1000 registros)
├── tests/
│   └── test_data.py              # Testes de validação do dataset
├── imgs/                         # Gráficos gerados
├── requirements.txt
└── .github/workflows/
    └── ci.yml                    # CI/CD automatizado
```

## Testes

```bash
pytest tests/ -v
# 6 testes: carregamento, colunas, target binário, scaling, train/test split
```

## Licença

MIT
