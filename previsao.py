from pyspark.sql import SparkSession
import joblib

spark = SparkSession.builder.appName("StressLevel_Previsao").getOrCreate()

df_pyspark = spark.read.csv(
    'datasets/dados_para_prever.csv', 
    header=True, 
    inferSchema=True
)

df_limpo = df_pyspark.dropna()

base = df_limpo.toPandas()

nome_arquivo = 'modelo_naive_bayes.joblib'
modelo = joblib.load(nome_arquivo)

previsoes = modelo.predict(base)

print("\nPrevisões geradas (classes previstas):")
print(previsoes)

spark.stop()