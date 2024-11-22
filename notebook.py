# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import col, when,count, avg, sum
from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, StringType
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, RobustScaler, StringIndexer, OneHotEncoder

# COMMAND ----------

# MAGIC %md
# MAGIC **1 Load and Clean Data**

# COMMAND ----------

# Load the dataset
dataset_path = "/mnt/landing/ml_use/telco_churn/landing/telco_customer_churn.csv"
input_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
target_col = "Churn"

# Drop irrelevant columns and preprocess
input_df = input_df.drop("Churn")  # Drop target column for now

# Convert columns to appropriate types
from pyspark.sql.types import BooleanType, DoubleType
from pyspark.sql.functions import when, col

# Convert Yes/No columns to Boolean
boolean_columns = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for column in boolean_columns:
    input_df = input_df.withColumn(column, when(col(column) == "Yes", True).otherwise(False))

# Convert numerical columns to Double
numerical_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
for column in numerical_columns:
    input_df = input_df.withColumn(column, col(column).cast(DoubleType()))

# Remove rows with invalid TotalCharges
processed_df = input_df.filter((col("TotalCharges") >= 0) | (col("TotalCharges").isNull()))

# Display processed data
display(processed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **2 Save Data to the Feature Store**

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

# Create a Feature Store Client
fs = FeatureStoreClient()

# Define the table name in Unity Catalog
catalog_name = "feyruz_hussen_yrih_da"  
schema_name = "default"  # Define schema
table_name = f"{catalog_name}.{schema_name}.telco_customer_features_for_pipeline"

# Create the feature table
fs.create_table(
    name=table_name,
    primary_keys=["customerID"],
    df=processed_df,
    description="Telco customer features for churn prediction",
    tags={"source": "processed_data", "format": "delta"}
)

# COMMAND ----------

# MAGIC %md
# MAGIC **3. Load Data from Feature Store**

# COMMAND ----------

# Read the data from the feature store
features_df = fs.read_table(table_name)
display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **4 Create Feature Engineering Pipeline**

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, RobustScaler, StringIndexer, OneHotEncoder

# Identify columns
string_cols = [c.name for c in features_df.schema.fields if c.dataType == StringType()]
num_cols = [c.name for c in features_df.schema.fields if c.dataType == DoubleType()]

# Impute missing values
imputer = Imputer(inputCols=num_cols, outputCols=num_cols, strategy='mean')

# Scale numerical columns
numerical_assembler = VectorAssembler(inputCols=num_cols, outputCol="numerical_assembled")
numerical_scaler = RobustScaler(inputCol="numerical_assembled", outputCol="numerical_scaled")

# Index and encode categorical columns
string_cols_indexed = [col + '_index' for col in string_cols]
string_indexer = StringIndexer(inputCols=string_cols, outputCols=string_cols_indexed, handleInvalid="keep")

ohe_cols = [col + '_ohe' for col in string_cols_indexed]
one_hot_encoder = OneHotEncoder(inputCols=string_cols_indexed, outputCols=ohe_cols, handleInvalid="keep")

# Assemble all features
feature_cols = ["numerical_scaled"] + ohe_cols
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Create pipeline
pipeline = Pipeline(stages=[
    imputer,
    numerical_assembler,
    numerical_scaler,
    string_indexer,
    one_hot_encoder,
    vector_assembler
])

# Fit the pipeline
pipeline_model = pipeline.fit(features_df)

# Transform the data
transformed_df = pipeline_model.transform(features_df)
display(transformed_df.select("features"))

# COMMAND ----------

# MAGIC %md
# MAGIC **5 Load and Save Pipeline**

# COMMAND ----------

import mlflow
import mlflow.spark

# Log the pipeline using MLflow
with mlflow.start_run() as run:
    mlflow.spark.log_model(pipeline_model, "feature_engineering_pipeline")
    print(f"Pipeline logged with run ID: {run.info.run_id}")

# Save the pipeline to a path
pipeline_model.write().overwrite().save("/dbfs/tmp/feature_engineering_pipeline")

# COMMAND ----------

# MAGIC %md
# MAGIC **- experiment 2: practice AutoML API/
# MAGIC - load data from feature store
# MAGIC - train classification model
# MAGIC - examine the model**
# MAGIC

# COMMAND ----------

table_name = f"{catalog_name}.{schema_name}.telco_customer_features_for_automl"

# COMMAND ----------

from databricks import automl
from datetime import datetime
automl_run = automl.classify(
    dataset = spark.table(table_name),
    target_col = "Churn",
    exclude_cols=["customerID"], # Exclude columns as needed
    timeout_minutes = 5)

# COMMAND ----------

import mlflow
# Get the experiment path by experiment ID
exp_path = mlflow.get_experiment(automl_run.experiment.experiment_id).name
# Find the most recent experiment in the AutoML folder
filter_string=f'name LIKE "{exp_path}"'
automl_experiment_id = mlflow.search_experiments(
  filter_string=filter_string,
  max_results=1,
  order_by=["last_update_time DESC"])[0].experiment_id

# COMMAND ----------

from mlflow.entities import ViewType

# Find the best run ...
automl_runs_pd = mlflow.search_runs(
  experiment_ids=[automl_experiment_id],
  filter_string=f"attributes.status = 'FINISHED'",
  run_view_type=ViewType.ACTIVE_ONLY,
  order_by=["metrics.val_f1_score DESC"]
)

# COMMAND ----------

print("best trial:" + str(automl_run.best_trial))

# COMMAND ----------

# Create the Destination path for storing the best run notebook
destination_path = "/Workspace/Repos/feyruz.hussen@cineplex.com/customer_churn/churn_best_run"

# Get the path and url for the generated notebook
result = automl.import_notebook(automl_run.trials[5].artifact_uri, destination_path)
print(f"The notebook is imported to: {result.path}")
print(f"The notebook URL           : {result.url}")
