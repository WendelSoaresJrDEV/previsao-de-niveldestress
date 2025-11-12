from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("StressLevel_LinearSVC_Pipeline").getOrCreate()

df_pyspark = spark.read.csv(
    'datasets/StressLevelDataset.csv', 
    header=True, 
    inferSchema=True
)

df_limpo = df_pyspark.dropna()

base = df_limpo.toPandas()
features = base.iloc[:, 0:20].values
classe = base.iloc[:, 20].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

previsores_teste, previsores_treinamento, classe_teste, classe_treinamento = train_test_split(features, classe, test_size=0.20)


classificador = LinearSVC(dual=False)
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

acuracia = accuracy_score(previsoes, classe_teste)
print("Acuracia do modelo: %.2f"%acuracia)

matriz = confusion_matrix(previsoes, classe_teste)
print(matriz)

modelo_nome = 'modelo_svc.joblib'
joblib.dump(classificador, modelo_nome)
print(f"Modelo salvo com o nome '{modelo_nome}'")

spark.stop()