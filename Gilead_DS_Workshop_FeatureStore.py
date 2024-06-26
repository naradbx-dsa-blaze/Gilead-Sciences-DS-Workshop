# Databricks notebook source
# MAGIC %md
# MAGIC ###FeatureStore Demo

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

file_path = '/Volumes/nara_catalog/gilead_ds_workshop/data/influenza_outbreak_dataset.mat'
X_train_df, y_train_df, X_holdout_df, y_holdout_df = load_influenza_outbreak(file_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Combine train and test dataframe to features

# COMMAND ----------

train = pd.concat([X_train_df, y_train_df], axis=1)
test = pd.concat([X_holdout_df, y_holdout_df], axis=1)

# Combine train and test dataframes
feature_df = pd.concat([train, test], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Rename the output column 

# COMMAND ----------

# Rename the last column
feature_df.columns.values[-1] = '545'

# COMMAND ----------

# Add a primary key column
feature_df['pkey'] = range(1, len(feature_df) + 1)
print(feature_df)

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
