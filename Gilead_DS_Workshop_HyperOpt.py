# Databricks notebook source
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

import numpy as np
import scipy.io
import pandas as pd
import numpy as np
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import time

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
# MAGIC ###Step 1: Tune Hyperparameters (XGBClassifier)
# MAGIC The XGBClassifier makes available a wide variety of hyperparameters which can be used to tune model training. Using some knowledge of our data and the algorithm, we might attempt to manually set some of the hyperparameters. But given the complexity of the interactions between them, it can be difficult to know exactly which combination of values will provide us the best model results. It's in scenarios such as these that we might perform a series of model runs with different hyperparameter settings to observe how the model responds and arrive at an optimal combination of values.
# MAGIC
# MAGIC Using hyperopt, we can automate this task, providing the hyperopt framework with a range of potential values to explore. Calling a function which trains the model and returns an evaluation metric, hyperopt can through the available search space to towards an optimum combination of values.
# MAGIC
# MAGIC For model evaluation, we will be using the average precision (AP) score which increases towards 1.0 as the model improves. Because hyperopt recognizes improvements as our evaluation metric declines, we will use -1 * the AP score as our loss metric within the framework.
# MAGIC
# MAGIC Putting this all together, we might arrive at model training and evaluation function as follows:

# COMMAND ----------

def evaluate_model(hyperopt_params): 
  
  # configure model parameters
  params = hyperopt_params
  
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  params['tree_method']='gpu_hist'      # settings for running on GPU
  params['predictor']='gpu_predictor'   # settings for running on GPU
  
  # instantiate model with parameters
  model = XGBClassifier(**params)
  
  # train
  model.fit(X_train_df, y_train_df)
  
  # predict
  y_prob = model.predict_proba(X_holdout_df)
  
  # score
  model_ap = average_precision_score(y_holdout_df, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)  # record actual metric with mlflow run
  
  # invert metric for hyperopt
  loss = -1 * model_ap  
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md The first part of the model evaluation function retrieves from memory replicated copies of our training and testing feature and label sets.  Our intent is to leverage SparkTrials in combination with hyperopt to parallelize the training of models across a Spark cluster, allowing us to perform multiple, simultaneous model training evaluation runs and reduce the overall time required to navigate the seach space.  By replicating our datasets to the worker nodes of the cluster, a task performed in the next cell, copies of the data needed for training and evaluation can be efficiently made available to the function with minimal networking overhead:
# MAGIC
# MAGIC **NOTE** See the Distributed Hyperopt [best practices documentation](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices.html#handle-datasets-of-different-orders-of-magnitude-notebook) for more options for data distribution.

# COMMAND ----------

# MAGIC %md The hyperparameter values delivered to the function by hyperopt are derived from a search space defined in the next cell.  Each hyperparameter in the search space is defined using an item in a dictionary, the name of which identifies the hyperparameter and the value of which defines a range of potential values for that parameter.  When defined using *hp.choice*, a parameter is selected from a predefined list of values.  When defined *hp.loguniform*, values are generated from a continuous range of values.  When defined using *hp.quniform*, values are generated from a continuous range but truncated to a level of precision identified by the third argument  in the range definition.  Hyperparameter search spaces in hyperopt may be defined in many other ways as indicated by the library's [online documentation](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions):  

# COMMAND ----------

# define minimum positive class scale factor
weights = compute_class_weight(
  'balanced', 
  classes=np.unique(y_train_df), 
  y=y_train_df
  )
scale = weights[1]/weights[0]

# define hyperopt search space
search_space = {
    'max_depth' : hp.quniform('max_depth', 1, 30, 1)                                  # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
    ,'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.40))     # learning rate for XGBoost
    ,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
    ,'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1)                   # minimum number of instances per node
    ,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
    ,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
    ,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
    ,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
    ,'scale_pos_weight' : hp.loguniform('scale_pos_weight', np.log(scale), np.log(scale * 10))   # weight to assign positive label to manage imbalance
    }

# COMMAND ----------

# MAGIC %md The remainder of the model evaluation function is fairly straightforward.  We simply train and evaluate our model and return our loss value, *i.e.* -1 * AP Score, as part of a dictionary interpretable by hyperopt.  Based on returned values, hyperopt will generate a new set of hyperparameter values from within the search space definition with which it will attempt to improve our metric. We will limit the number of hyperopt evaluations to 250 simply based on a few trail runs we performed (not shown).  The larger the potential search space and the degree to which the model (in combination with the training dataset) responds to different hyperparameter combinations determines how many iterations are required for hyperopt to arrive at locally optimal values.  You can examine the output of the hyperopt run to see how our loss metric slowly improves over the course of each of these evaluations:
# MAGIC
# MAGIC **NOTE** The XGBClassifier is configured within the *evaluate_model* function to use **GPUs**. Make sure you are running this on a **GPU-based cluster**.

# COMMAND ----------

import mlflow

experiment_name = "/Workspace/Users/narasimha.kamathardi@databricks.com/Gilead Sciences DS Workshop/Gilead_Hyperopt_ML_Experiment"

# Check if the experiment exists
existing_experiment = mlflow.get_experiment_by_name(experiment_name)
if existing_experiment:
    # Delete the existing experiment
    mlflow.delete_experiment(existing_experiment.experiment_id)

# Set the experiment
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    argmin = fmin(
        fn=evaluate_model,
        space=search_space,
        algo=tpe.suggest,
        max_evals=250,
        trials=SparkTrials(parallelism=4),
        verbose=True
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train model with optimal settings 

# COMMAND ----------

# train model with optimal settings 
with mlflow.start_run(run_name='XGB Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
   
  # configure params
  params = space_eval(search_space, argmin)
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])
  if 'scale_pos_weight' in params: params['scale_pos_weight']=int(params['scale_pos_weight'])    
  params['tree_method']='hist'        # modified for CPU deployment
  params['predictor']='cpu_predictor' # modified for CPU deployment
  mlflow.log_params(params)
  
  # train
  model = XGBClassifier(**params)
  model.fit(X_train_df, y_train_df)
  mlflow.sklearn.log_model(model, 'model')  # persist model with mlflow
  
  # predict
  y_prob = model.predict_proba(X_holdout_df)
  
  # score
  model_ap = average_precision_score(y_holdout_df, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))
