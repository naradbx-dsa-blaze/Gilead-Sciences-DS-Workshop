# Databricks notebook source
# MAGIC %md
# MAGIC ### Copy LR model artifacts to workspace folder

# COMMAND ----------

import mlflow 

model_name = "nara_catalog.gilead_ds_workshop.lr_influenza_outbreak"
version = 1

mlflow.set_registry_uri("databricks-uc") 
model_uri = f"models:/{model_name}/{version}" # reference model by version or alias 
destination_path = "/Workspace/Users/narasimha.kamathardi@databricks.com/Gilead Sciences DS Workshop/Gilead-Sciences-DS-Workshop/lr_influenza_outbreak" 

mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=destination_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copy RF model artifacts to workspace folder

# COMMAND ----------

model_name = "nara_catalog.gilead_ds_workshop.rf_influenza_outbreak"
version = 1

mlflow.set_registry_uri("databricks-uc") 
model_uri = f"models:/{model_name}/{version}" # reference model by version or alias 
destination_path = "/Workspace/Users/narasimha.kamathardi@databricks.com/Gilead Sciences DS Workshop/Gilead-Sciences-DS-Workshop/rf_influenza_outbreak" 

mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=destination_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copy XGB model artifacts to workspace folder

# COMMAND ----------

model_name = "nara_catalog.gilead_ds_workshop.xgb_influenza_outbreak"
version = 1

mlflow.set_registry_uri("databricks-uc") 
model_uri = f"models:/{model_name}/{version}" # reference model by version or alias 
destination_path = "/Workspace/Users/narasimha.kamathardi@databricks.com/Gilead Sciences DS Workshop/Gilead-Sciences-DS-Workshop/xgb_influenza_outbreak" 

mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=destination_path)
