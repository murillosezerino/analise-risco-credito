
# Análise de Risco de Crédito

# 1. Importação das Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 2. Carregamento dos Dados
df = pd.read_csv("dados_credito.csv")

# 3. Pré-processamento dos Dados
# Convert categorical variables to numeric using Label Encoding
le = LabelEncoder()
categorical_columns = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']
for column in categorical_columns:
    df[column] = le.fit_transform(df[column].astype(str))

# Fill missing values with median for numeric columns
numeric_columns = ['Age', 'Job', 'Credit amount', 'Duration']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Padronização das variáveis numéricas
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 4. Análise de Balanceamento da variável alvo
print("\nDistribuição da variável 'Risk':")
print(df['Risk'].value_counts(normalize=True))

# 5. Seleção de Variáveis
X = df[['Age', 'Job', 'Credit amount', 'Duration', 'Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']]
y = df['Risk']

# 6. Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Treinamento dos Modelos
modelos = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Logistic Regression': LogisticRegression()
}

resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    resultados[nome] = {
        'modelo': modelo,
        'acuracia': acc,
        'auc': auc,
        'pred': y_pred
    }

# 8. Avaliação do Melhor Modelo (Random Forest)
print(f"\nMelhor modelo: Random Forest")
print(f"Acurácia: {resultados['Random Forest']['acuracia']:.2f}")
print(f"AUC: {resultados['Random Forest']['auc']:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, resultados['Random Forest']['pred']))

# Matriz de Confusão
cm = confusion_matrix(y_test, resultados['Random Forest']['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Random Forest')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# 9. Curva ROC dos modelos
plt.figure(figsize=(8, 6))
for nome, res in resultados.items():
    RocCurveDisplay.from_estimator(res['modelo'], X_test, y_test, name=nome, ax=plt.gca())
plt.title("Curva ROC - Comparação entre Modelos")
plt.grid()
plt.show()

# 10. Validação Cruzada com Random Forest
scores_cv = cross_val_score(resultados['Random Forest']['modelo'], X, y, cv=5, scoring='accuracy')
print(f"\nAcurácia média com validação cruzada (Random Forest): {scores_cv.mean():.2f}")

# 11. Visualização da Importância das Variáveis
importances = resultados['Random Forest']['modelo'].feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(X.columns, importances)
plt.title("Importância das Features - Random Forest")
plt.ylabel("Score")
plt.xlabel("Variáveis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 12. Comparação entre os modelos – Acurácia e AUC
metricas_df = pd.DataFrame({
    'Modelo': list(resultados.keys()),
    'Acurácia': [res['acuracia'] for res in resultados.values()],
    'AUC': [res['auc'] for res in resultados.values()]
})

# Plotando as métricas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=metricas_df, x='Modelo', y='Acurácia', ax=axes[0])
axes[0].set_title('Comparação de Acurácia')
axes[0].set_ylim(0, 1)

sns.barplot(data=metricas_df, x='Modelo', y='AUC', ax=axes[1])
axes[1].set_title('Comparação de AUC')
axes[1].set_ylim(0, 1)

plt.suptitle('Desempenho dos Modelos de Classificação')
plt.tight_layout()
plt.show()
