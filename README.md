Análise de Risco de Crédito

Este projeto realiza uma análise de risco de crédito utilizando técnicas de aprendizado de máquina. O objetivo é prever o risco associado a clientes com base em seus dados financeiros e pessoais.

Sobre o Projeto

A análise inclui:

- Pré-processamento de dados (limpeza, codificação, normalização)
- Treinamento de três modelos: Random Forest, Gradient Boosting e Regressão Logística
- Avaliação de desempenho com métricas de acurácia, AUC e matriz de confusão
- Visualização da curva ROC e da importância das variáveis
- Validação cruzada com o melhor modelo

Tecnologias Utilizadas

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Estrutura do Projeto

```
analise-risco-credito/
├── dados/                 # Arquivo CSV com os dados brutos
├── src/                   # Código-fonte da análise
├── imgs/                  # Gráficos e imagens geradas
├── README.md              # Documentação
├── requirements.txt       # Bibliotecas utilizadas
└── .gitignore             # Arquivos ignorados pelo Git
```

Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/analise-risco-credito.git
cd analise-risco-credito
```

2. Crie um ambiente virtual e ative:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute o script:
```bash
python src/analise_risco_credito.py
```

Exemplos de Saída

As seguintes visualizações são geradas:

- Matriz de Confusão
- Curva ROC para comparação entre modelos
- Importância das variáveis
- Comparativo de métricas (Acurácia e AUC)

Contato
[Murillo Sezerino] – murillosze@gmail.com
