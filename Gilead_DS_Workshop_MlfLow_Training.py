# Databricks notebook source
# MAGIC %md
# MAGIC ###Use Case Description:
# MAGIC - The data is from the United States. The data comes from different states under different weeks. For each week, the task is to predict whether or not there is an influenza outbreak on the next date. More specifically, for influenza activity, there are four levels of flu activities from minimal to high according to CDC Flu Activity Map. An influenza outbreak occurrence is indicated if the activity level is high.

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]"
# MAGIC %pip install xgboost
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#import necessary library
import scipy.io
import pandas as pd
import numpy as np
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
from mlflow import sklearn

# COMMAND ----------

# MAGIC %md
# MAGIC ###Dataset Information:
# MAGIC - The input of the prediction task is the set of the keyword counts for all the tweets in a state in a week.
# MAGIC - The output is the occurrence of influenza outbreak for the specific state in the next week, which is zero if no event in the next week; or one, otherwise. Here are the briefs of all the variables-
# MAGIC     - flu_locations': a list of states.
# MAGIC     - 'flu_keywords': keyword list.
# MAGIC     - 'flu_X_*': input data for all the locations and all the weeks.
# MAGIC     - 'flu_Y_*': output data for all the locations and all the weeks.
# MAGIC
# MAGIC 525 keywords specified in the variable 'flu_keywords' in the data

# COMMAND ----------

# MAGIC %md
# MAGIC ###Preprocessing Twitter Data

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
# MAGIC ###Training features on logistic regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with mlflow.start_run():
  model = LogisticRegression(max_iter=1000)
  model.fit(X_train_df, y_train_df)
  y_pred = model.predict(X_holdout_df)
  # Take the first row of the training dataset as the model input example.
  input_example = X_train_df.iloc[[0]] 
  mlflow.sklearn.log_model( 
        sk_model=model, 
        artifact_path="lr_influenza_outbreak", 
        # The signature is automatically inferred from the input example and its predicted output. 
        input_example=input_example, 
        registered_model_name="nara_catalog.gilead_ds_workshop.lr_influenza_outbreak", 
    )
# Evaluate the model
accuracy = accuracy_score(y_holdout_df, y_pred)
print("Accuracy:", accuracy)
# Log the accuracy metric to MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Training features on random forest

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with mlflow.start_run():
    # Initialize the random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train_df, y_train_df)
    
    # Predict on the holdout set
    y_pred = model.predict(X_holdout_df)
    
    # Take the first row of the training dataset as the model input example.
    input_example = X_train_df.iloc[[0]]
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="rf_influenza_outbreak", 
        # The signature is automatically inferred from the input example and its predicted output. 
        input_example=input_example, 
        registered_model_name="nara_catalog.gilead_ds_workshop.rf_influenza_outbreak", 
    )
    
    # Evaluate the model
    accuracy = accuracy_score(y_holdout_df, y_pred)
    print("Accuracy:", accuracy)
    # Log the accuracy metric to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Training features on xgboost classifier

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from mlflow import xgboost

with mlflow.start_run():
    # Initialize the XGBoost model
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Train the model
    model.fit(X_train_df, y_train_df)
    
    # Predict on the holdout set
    y_pred = model.predict(X_holdout_df)
    
    # Take the first row of the training dataset as the model input example.
    input_example = X_train_df.iloc[[0]]
    
    # Log the model to MLflow
    mlflow.xgboost.log_model(
        xgb_model=model,  # Change booster=model to xgb_model=model
        artifact_path="xgb_influenza_outbreak", 
        # The signature is automatically inferred from the input example and its predicted output. 
        input_example=input_example, 
        registered_model_name="nara_catalog.gilead_ds_workshop.xgb_influenza_outbreak", 
    )
    
    # Evaluate the model
    accuracy = accuracy_score(y_holdout_df, y_pred)
    print("Accuracy:", accuracy)
    # Log the accuracy metric to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.end_run()
