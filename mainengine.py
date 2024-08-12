# Autor: Gabriel André
# Universidade do Estado do Rio de Janeiro - UERJ-ZO
import pandas as pd
import sns
from matplotlib import pyplot as plt

# Carregando o arquivo CSV
df = pd.read_csv('motor_data_simplificada.csv')

# Verificando as correlações entre as variáveis
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap de Correlação')
#plt.show()

# Visualizando a Distribuição das variáveis numéricas
df.hist(bins=30, figsize=(15, 10), color='blue')
plt.suptitle('Distribuição das Variáveis')
#plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definindo as variáveis dependentes e independentes
X = df.drop('estado_motor', axis=1)
y = df['estado_motor']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizando as variáveis
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Criar o modelo
log_model = LogisticRegression()

# Treinar o modelo
log_model.fit(X_train, y_train)

# Fazer previsões
y_pred = log_model.predict(X_test)
y_proba = log_model.predict_proba(X_test)[:, 1]

# Avaliar o modelo
print(classification_report(y_test, y_pred))
print("AUC: ", roc_auc_score(y_test, y_proba))

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues")
plt.title('Matriz de Confusão')
plt.show()

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
classes = ['0', '1', 'Macro Avg', 'Weighted Avg']
precision = [0.98, 0.69, 0.83, 0.96]
recall = [0.99, 0.56, 0.77, 0.96]
f1_score = [0.98, 0.62, 0.80, 0.96]

# Posicionamento das barras
x = np.arange(len(classes))
width = 0.25  # Largura das barras

# Criar a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Plotar as barras
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Adicionar algumas informações
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1-Score by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Função para adicionar rótulos nas barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 pontos verticais offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Adicionar rótulos
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

# Mostrar o gráfico
plt.show()

