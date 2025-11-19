import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("StressLevel_NaiveBayes_Pipeline").getOrCreate()

#Treinamento do modelo
df_treinamento = spark.read.csv(
    'datasets/StressLevelDataset_ParaTreinamento', 
    header=True, 
    inferSchema=True
)

df_limpo_treino = df_treinamento.dropna()
base_treino = df_limpo_treino.toPandas()

features_treinamento = base_treino.iloc[:, 0:20].values
classe_treinamento = base_treino.iloc[:, 20].values

labelencoder = LabelEncoder()
classe_treinamento = labelencoder.fit_transform(classe_treinamento)

classificador = GaussianNB(priors=None, var_smoothing=1e-9)
classificador.fit(features_treinamento, classe_treinamento)

modelo_nome = 'modelo_naive_bayes.joblib'
joblib.dump(classificador, modelo_nome)
print(f"Modelo salvo com o nome '{modelo_nome}'")

#Teste do modelo
modelo = joblib.load(modelo_nome)

df_teste = spark.read.csv(
    'datasets/StressLevelDataset_ParaTeste', 
    header=True, 
    inferSchema=True
)

df_limpo_teste = df_teste.dropna()
base_teste = df_limpo_teste.toPandas()

features_teste = base_teste.iloc[:, 0:20].values
classe_teste = base_teste.iloc[:, 20].values

classe_teste = labelencoder.fit_transform(classe_teste)

previsoes = classificador.predict(features_teste)

acuracia = accuracy_score(previsoes, classe_teste)
print("Acuracia do modelo: %.2f"%acuracia)

matriz = confusion_matrix(previsoes, classe_teste)
print(matriz)

#Salvar resultados em um CSV
classes_reais_texto = labelencoder.inverse_transform(classe_teste) 
previsoes_texto = labelencoder.inverse_transform(previsoes)

feature_columns = base_teste.columns[:features_teste.shape[1]]
df_resultado = pd.DataFrame(features_teste, columns=feature_columns)

df_resultado['stress_level'] = classes_reais_texto
df_resultado['Previsão do Modelo Naive Bayes'] = previsoes_texto

nome_arquivo_saida = 'resultados_previsao_naive_bayes.csv'
df_resultado.to_csv(nome_arquivo_saida, index=False) 

print(f"\nResultados salvos com sucesso em '{nome_arquivo_saida}'")

spark.stop()