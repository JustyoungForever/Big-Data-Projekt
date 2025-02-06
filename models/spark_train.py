import time
import os
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, expm1
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

start_time = time.time()

spark = SparkSession.builder \
    .appName("Hadoop-ML-Salary-Prediction") \
    .master("yarn") \
    .config("spark.executor.instances", "3") \
    .config("spark.executor.cores", "1") \
    .config("spark.executor.memory", "1g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.yarn.am.memory", "1g") \
    .getOrCreate()

HDFS_NAMENODE = "hdfs://node1:8020"
hdfs_data_path = f"{HDFS_NAMENODE}/user/hadoop/datasets/cleaned_data"
hdfs_results_dir = f"{HDFS_NAMENODE}/user/hadoop/results/"

hadoop_conf = spark._jsc.hadoopConfiguration()
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_results_dir)
if not fs.exists(path):
    fs.mkdirs(path)

# Load datasets
df_ai = spark.read.csv(f"{hdfs_data_path}/cleaned_global_ai_ml_data_salaries.csv", header=True, inferSchema=True)
df_ai_ml = spark.read.csv(f"{hdfs_data_path}/cleaned_global-salaries-in-ai-ml-data-science.csv", header=True, inferSchema=True)
df_jobs_2024 = spark.read.csv(f"{hdfs_data_path}/cleaned_jobs_in_data_2024.csv", header=True, inferSchema=True)
df_data_science = spark.read.csv(f"{hdfs_data_path}/cleaned_data_science_salaries.csv", header=True, inferSchema=True)

datasets = {
    "df_ai": df_ai,
    "df_ai_ml": df_ai_ml,
    "df_jobs_2024": df_jobs_2024,
    "df_data_science": df_data_science
}

# Process datasets
target_variable = "salary_in_usd"
processed_datasets = {}
for name, df in datasets.items():
    available_features = [col for col in df.columns if col in ["work_year", "experience_level", "employment_type", "company_size", "job_title"]]
    if "job_title" in df.columns:
        job_title_indexer = StringIndexer(inputCol="job_title", outputCol="job_title_encoded")
        df = job_title_indexer.fit(df).transform(df)
        available_features.remove("job_title")
        available_features.append("job_title_encoded")
    if target_variable in df.columns:
        processed_datasets[name] = df.select(*available_features, target_variable)

combined_data = None
for df in processed_datasets.values():
    if combined_data is None:
        combined_data = df
    else:
        combined_data = combined_data.unionByName(df, allowMissingColumns=True)

# Process categorical features
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
categorical_columns = ["experience_level", "employment_type", "company_size"]
if "job_title_encoded" in combined_data.columns:
    categorical_columns.append("job_title_encoded")

indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_columns]
onehot_encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_ohe") for col in categorical_columns]
feature_columns = ["work_year"] + [col + "_ohe" for col in categorical_columns]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
pipeline = Pipeline(stages=indexers + onehot_encoders + [assembler])
processed_data = pipeline.fit(combined_data).transform(combined_data)
final_data = processed_data.select("features", "salary_in_usd")

# Split data
train, test = final_data.randomSplit([0.8, 0.2], seed=42)
train = train.repartition(6)
test = test.repartition(6)

train.cache()
test.cache()
# Train models
def train_and_evaluate_model(model, train, test, model_name):
    start_time = time.time()
    trained_model = model.fit(train)
    predictions = trained_model.transform(test)
    evaluator = RegressionEvaluator(labelCol="salary_in_usd", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol="salary_in_usd", predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="salary_in_usd", predictionCol="prediction", metricName="r2").evaluate(predictions)
    end_time = time.time()
    print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f} (Time: {end_time - start_time:.2f}s)")
    return {"model_name": model_name, "rmse": rmse, "mae": mae, "r2": r2, "trained_model": trained_model}

rf = RandomForestRegressor(featuresCol="features", labelCol="salary_in_usd", numTrees=100, seed=42, maxBins=250)
rf_results = train_and_evaluate_model(rf, train, test, "Random Forest")

gbt = GBTRegressor(featuresCol="features", labelCol="salary_in_usd", maxIter=100, seed=42)
gbt_results = train_and_evaluate_model(gbt, train, test, "Gradient Boosting")

lr = LinearRegression(featuresCol="features", labelCol="salary_in_usd")
lr_results = train_and_evaluate_model(lr, train, test, "Linear Regression")

# Save results
results_df = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "Linear Regression"],
    "RMSE": [rf_results["rmse"], gbt_results["rmse"], lr_results["rmse"]],
    "MAE": [rf_results["mae"], gbt_results["mae"], lr_results["mae"]],
    "R²": [rf_results["r2"], gbt_results["r2"], lr_results["r2"]]
})

plt.figure(figsize=(12, 6))
results_df_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
sns.barplot(data=results_df_melted, x="Model", y="Value", hue="Metric")
plt.title("Model Performance Comparison")
plt.ylabel("Error Value")
plt.xlabel("Model")
plt.xticks(rotation=45)

local_results_dir = "/home/hadoop/results/"
os.makedirs(local_results_dir, exist_ok=True)
local_results_image = "/tmp/spark_ml_results.png"
hdfs_results_image = f"{hdfs_results_dir}/spark_ml_results.png"
plt.savefig(local_results_image)

fs.copyFromLocalFile(False, True, spark._jvm.org.apache.hadoop.fs.Path(local_results_image),
                     spark._jvm.org.apache.hadoop.fs.Path(hdfs_results_image))
for file_status in fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(hdfs_results_dir)):
    file_path = file_status.getPath().toString()
    file_name = file_path.split("/")[-1]
    local_file_path = os.path.join(local_results_dir, file_name)
    
    print(f"Downloading {file_path} to {local_file_path}")
    fs.copyToLocalFile(False, spark._jvm.org.apache.hadoop.fs.Path(file_path),
                        spark._jvm.org.apache.hadoop.fs.Path(local_file_path))
print(f"Image saved to HDFS: {hdfs_results_image}")
spark.stop()
print("Training completed.")


