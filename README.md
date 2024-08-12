# Aplicação de Regressão Logística para o Supervisionamento Inteligente de Motores

Este repositório contém o código e a análise referentes ao artigo "Aplicação de Regressão Logística para o Supervisionamento Inteligente de Motores". O trabalho faz parte do projeto de Capacitação Técnica para Automação Industrial e Acionamento de Motores da Universidade do Estado do Rio de Janeiro - Zona Oeste.

## Descrição

O artigo explora a aplicação de regressão logística para o monitoramento e previsão de falhas em motores. Utiliza dados simulados e modelos estatísticos para avaliar o desempenho da técnica na identificação de problemas em motores.

## Estrutura do Repositório

- **`motor_data_simplificada.csv`**: Arquivo CSV com dados de motores utilizados na análise.
- **`regressao_logistica.py`**: Código Python para a aplicação de regressão logística, incluindo análise exploratória dos dados, treinamento do modelo e avaliação dos resultados.
- **`README.md`**: Este arquivo.

## Dependências

O código foi desenvolvido utilizando Python e requer as seguintes bibliotecas:

- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Para instalar essas dependências, execute o comando:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

## Uso

1. **Carregar e Preparar os Dados**

   O primeiro passo é carregar os dados e realizar a análise exploratória para entender as variáveis e suas correlações.

   ```python
   import pandas as pd
   import seaborn as sns
   from matplotlib import pyplot as plt

   # Carregando o arquivo CSV
   df = pd.read_csv('motor_data_simplificada.csv')

   # Verificando as correlações entre as variáveis
   plt.figure(figsize=(10, 8))
   sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
   plt.title('Heatmap de Correlação')
   plt.show()

   # Visualizando a Distribuição das variáveis numéricas
   df.hist(bins=30, figsize=(15, 10), color='blue')
   plt.suptitle('Distribuição das Variáveis')
   plt.show()
   ```
2. **Treinamento do Modelo**
   
    O próximo passo é preparar os dados para o treinamento do modelo e ajustar a regressão logística.
  
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    
    # Definindo as variáveis dependentes e independentes
    X = df.drop('estado_motor', axis=1)
    y = df['estado_motor']
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Padronizando as variáveis
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
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
    ```
3. **Análise dos Resultados**
   
    Além das métricas principais, é importante analisar a precisão, recall e f1-score para cada classe.
    
    ```python
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
    ```
# Imagens obtidas após análises
![image](https://github.com/user-attachments/assets/4dfb6f2d-f83f-41d2-9b7f-7403ecc2af2f)
![image](https://github.com/user-attachments/assets/d6e13605-1855-4e71-b1f4-8073a3adeef0)
![image](https://github.com/user-attachments/assets/1357e95f-f174-43b0-a79e-0d3ff142c483)


# Contribuições
Sinta-se à vontade para contribuir com melhorias e correções. As contribuições são bem-vindas através de pull requests ou issues.

# Contato
Para qualquer dúvida ou sugestão, você pode me contatar em:

Nome: Gabriel André de Lima Silva
Instituição: Universidade do Estado do Rio de Janeiro - Zona Oeste
Projeto: Capacitação Técnica para Automação Industrial e Acionamento de Motores
