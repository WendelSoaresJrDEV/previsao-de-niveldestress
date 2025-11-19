from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("QuebrarCSV").getOrCreate()

df = spark.read.csv(
    'datasets/StressLevelDataset.csv', 
    header=True, 
    inferSchema=True
)

df_treinamento, df_teste = df.randomSplit([0.85, 0.15])

print(f"Separação bem sucedida, salvando arquivos")

df_treinamento.write.csv("datasets/StressLevelDataset_ParaTreinamento", header = True)
df_teste.write.csv("datasets/StressLevelDataset_ParaTeste", header = True)

print("Arquivos salvos")

spark.stop()