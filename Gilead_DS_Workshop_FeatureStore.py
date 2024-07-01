# Databricks notebook source
# MAGIC %md
# MAGIC # FeatureStore Demo
# MAGIC
# MAGIC %md
# MAGIC
# MAGIC # Getting started with Feature Engineering in Databricks Unity Catalog
# MAGIC
# MAGIC The <a href="https://docs.databricks.com/en/machine-learning/feature-store/uc/feature-tables-uc.html" target="_blank">Feature Engineering in Databricks Unity Catalog</a> allows you to create a centralized repository of features. These features can be used to train & call your ML models. By saving features as feature engineering tables in Unity Catalog, you will be able to:
# MAGIC
# MAGIC - Share features across your organization 
# MAGIC - Increase discoverability sharing 
# MAGIC - Ensures that the same feature computation code is used for model training and inference
# MAGIC - Enable real-time backend, leveraging your Delta Lake tables for batch training and Key-Value store for realtime inferences
# MAGIC
# MAGIC ## Demo content
# MAGIC
# MAGIC Multiple version of this demo are available, each version introducing a new concept and capabilities. We recommend following them 1 by 1.
# MAGIC
# MAGIC ### Introduction (this notebook)
# MAGIC
# MAGIC  - Ingest our data and save them as a feature table within Unity Catalog
# MAGIC  - Create a Feature Lookup with multiple tables
# MAGIC  - Train your model using the Feature Engineering Client
# MAGIC  - Register your best model and promote it into Production
# MAGIC  - Perform batch scoring
# MAGIC
# MAGIC ### Advanced version ([open the notebook]($./02_Feature_store_advanced))
# MAGIC
# MAGIC  - Join multiple Feature Store tables
# MAGIC  - Point in time lookup
# MAGIC  - Online tables
# MAGIC
# MAGIC ### Expert version ([open the notebook]($./03_Feature_store_expert))
# MAGIC  - Streaming Feature Store tables 
# MAGIC  - Feature spec (with functions) saved in UC 
# MAGIC  - Feature spec endpoint to compute inference features in realtime (like distance)
# MAGIC
# MAGIC  
# MAGIC *For more detail on the Feature Engineering in Unity Catalog, open <a href="https://api-docs.databricks.com/python/feature-engineering/latest" target="_blank">the documentation</a>.*

# COMMAND ----------

#install feature engineering package
%pip install databricks-feature-engineering
dbutils.library.restartPython()

# COMMAND ----------

#install libraries
import scipy.io
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
from mlflow import sklearn

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 1: Create our Feature Engineering table
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/feature_store/feature_store_creation.png" alt="Feature Engineering Table Creation" width="500px" style="margin-left: 10px; float: right"/>
# MAGIC
# MAGIC Our first step is to create our Feature Engineering table.
# MAGIC
# MAGIC We will load data from the silver table `travel_purchase` and create features from these values. 
# MAGIC
# MAGIC In this first version, we'll transform the timestamp into multiple features that our model will be able to understand. 
# MAGIC
# MAGIC In addition, we will drop the label from the table as we don't want it to leak our features when we do our training.
# MAGIC
# MAGIC To create the feature table, we'll use the `FeatureEngineeringClient.create_table`. 
# MAGIC
# MAGIC Under the hood, this will create a Delta Table to save our information. 
# MAGIC
# MAGIC These steps would typically live in a separate job that we call to refresh our features when new data lands in the silver table.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Function to clean the influenza data and vectorize inout and output columns

# COMMAND ----------

def load_influenza_outbreak(file_path, holdout=True):
    mat = scipy.io.loadmat(file_path)
    flu_x_tr = mat["flu_X_tr"]
    flu_x_te = mat["flu_X_te"]
    flu_y_tr = mat["flu_Y_tr"]
    flu_y_te = mat["flu_Y_te"]

    x_tr = vstack([flu_x_tr[0,i] for i in range(flu_x_tr.shape[1])])
    x_te = vstack([flu_x_te[0,i] for i in range(flu_x_te.shape[1])])
    x_arr = vstack([x_tr, x_te]).toarray()
    x_df = pd.DataFrame(x_arr)

    y_tr = np.concatenate([flu_y_tr[0,i] for i in range(flu_y_tr.shape[1])])
    y_te = np.concatenate([flu_y_te[0,i] for i in range(flu_y_te.shape[1])])
    y_arr = np.concatenate([y_tr, y_te])

    y_df = pd.Series(LabelEncoder().fit_transform(y_arr))
    X_train_df, X_holdout_df, y_train_df, y_holdout_df = train_test_split(x_df, y_df, test_size=0.2, random_state=42, stratify=y_df)
    X_train_df.reset_index(drop=True, inplace=True)
    y_train_df.reset_index(drop=True, inplace=True)

    if holdout:
        print("X, y Holdout shapes: ", X_holdout_df.shape, y_holdout_df.shape)
        return X_train_df, y_train_df, X_holdout_df, y_holdout_df
    else:
        return X_train_df, y_train_df

# COMMAND ----------

#call the function to partition the datasets to train and test sets
file_path = '/Volumes/nara_catalog/gilead_ds_workshop/data/influenza_outbreak_dataset.mat'
X_train_df, y_train_df, X_holdout_df, y_holdout_df = load_influenza_outbreak(file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Combine train and test dataframe to features

# COMMAND ----------

train = pd.concat([X_train_df, y_train_df], axis=1)
test = pd.concat([X_holdout_df, y_holdout_df], axis=1)

# Combine train and test dataframes to a single dataframe
feature_df = pd.concat([train, test], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Rename the output column 

# COMMAND ----------

# Rename the last column
feature_df.columns.values[-1] = '545'

# COMMAND ----------

# Add a primary key column as its mandatory for feature store tables
feature_df['pkey'] = range(1, len(feature_df) + 1)
print(feature_df)

# COMMAND ----------

#convert to spark dataframe as pandas dataframes are not supported
feature_df = spark.createDataFrame(feature_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create feature table

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create feature table with `pkey` as the primary key.
# Take schema from DataFrame output by compute_customer_features
customer_feature_table = fe.create_table(
  name='nara_catalog.gilead_ds_workshop.influenza_outbreak_features',
  primary_keys='pkey',
  schema=feature_df.schema,
  description='influenza outbreak encoded features'
)

# COMMAND ----------

#write data to feature store table
fe.write_table(
    name='nara_catalog.gilead_ds_workshop.influenza_outbreak_features',
    df = feature_df
)
