# Databricks notebook source
# MAGIC %md
# MAGIC # FeatureStore Demo
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
# MAGIC ### Introduction (this notebook)
# MAGIC
# MAGIC  - Ingest our data and save them as a feature table within Unity Catalog
# MAGIC  
# MAGIC *For more detail on the Feature Engineering in Unity Catalog, open <a href="https://docs.databricks.com/en/machine-learning/feature-store/index.html" target="_blank">the documentation</a>.*

# COMMAND ----------

#install feature engineering package
%pip install databricks-feature-engineering
dbutils.library.restartPython()

# COMMAND ----------

#Import libraries
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

# MAGIC %md
# MAGIC ###Function to clean the influenza data and vectorize input and output columns

# COMMAND ----------

file_path = '/Volumes/nara_catalog/gilead_ds_workshop/data/influenza_outbreak_dataset.mat'
mat = scipy.io.loadmat(file_path)
print(mat.keys())
flu_x_tr = mat["flu_X_tr"]
flu_x_te = mat["flu_X_te"]
flu_y_tr = mat["flu_Y_tr"]
flu_y_te = mat["flu_Y_te"]

# COMMAND ----------

flu_x_tr = mat["flu_X_tr"]
print(flu_x_tr.shape)

# COMMAND ----------

x_tr = vstack([flu_x_tr[0,i] for i in range(flu_x_tr.shape[1])])
x_te = vstack([flu_x_te[0,i] for i in range(flu_x_te.shape[1])])
x_arr = vstack([x_tr, x_te]).toarray()
x_df = pd.DataFrame(x_arr)

y_tr = np.concatenate([flu_y_tr[0,i] for i in range(flu_y_tr.shape[1])])
y_te = np.concatenate([flu_y_te[0,i] for i in range(flu_y_te.shape[1])])
y_arr = np.concatenate([y_tr, y_te])
y_df = pd.DataFrame(y_arr)

# COMMAND ----------

for i in range(flu_x_tr.shape[1]):
  print(flu_x_tr[0,i])

# COMMAND ----------

for i in range(x_tr.shape[1]):
  print(x_tr[:,i])

# COMMAND ----------

x_tr.shape

# COMMAND ----------

y_df.head()

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

x_arr.shape

# COMMAND ----------

#call the function to partition the datasets to train and test sets
file_path = '/Volumes/nara_catalog/gilead_ds_workshop/data/influenza_outbreak_dataset.mat'
X_train_df, y_train_df, X_holdout_df, y_holdout_df = load_influenza_outbreak(file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Combine train and test dataframe to create a features store

# COMMAND ----------

train = pd.concat([X_train_df, y_train_df], axis=1)
test = pd.concat([X_holdout_df, y_holdout_df], axis=1)

# Combine train and test dataframes to a single dataframe
feature_df = pd.concat([train, test], axis=0) # axis=0 for rows, axis=1 for columns

# COMMAND ----------

X_train_df.shape, y_train_df.shape, X_holdout_df.shape, y_holdout_df.shape

# COMMAND ----------

train.shape, test.shape, feature_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ###Rename the output column 

# COMMAND ----------

# Rename the last column
feature_df.columns.values[-1] = '545' # last column after the concatenation of X, Y, is renamed to 545

# COMMAND ----------

# Add a primary key column as its mandatory for feature store tables
feature_df['pkey'] = range(1, len(feature_df) + 1) # To add primary key to each of the 75840 rows
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
    df = feature_df, mode='merge'
)

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
