# Databricks notebook source
# MAGIC %md
# MAGIC # Managing the model lifecycle in Unity Catalog
# MAGIC <p>
# MAGIC <img src="https://github.com/naradbx-dsa-blaze/Gilead-Sciences-DS-and-DE-Workshop/blob/feature/ddavis_model_registry/model-registry/img/ml-lifecycle.png?raw=true" width="1000">
# MAGIC
# MAGIC **Common Pain Points without model registry:**
# MAGIC - Models are scattered
# MAGIC - Limited visibility into versions and version history
# MAGIC - Limited visibility into model lifecycle (e.g. staging, production, archived) 
# MAGIC
# MAGIC ## **Models in Unity Catalog**
# MAGIC <p>
# MAGIC <img src="https://github.com/naradbx-dsa-blaze/Gilead-Sciences-DS-and-DE-Workshop/blob/feature/ddavis_model_registry/model-registry/img/uc-models.png?raw=true" width="800">
# MAGIC
# MAGIC [Models in Unity Catalog](https://docs.databricks.com/en/mlflow/models-in-uc.html) addresses this challenge and enables members of the data team to:
# MAGIC <br>
# MAGIC * **Discover** registered models, current aliases in model development, experiment runs, and associated code with a registered model
# MAGIC * **Promote** models to different phases of their lifecycle with the use of model aliases
# MAGIC * **Tag** models to capture metadata specific to your MLOps process
# MAGIC * **Deploy** different versions of a registered model, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * **Test** models in an automated fashion
# MAGIC * **Document** models throughout their lifecycle
# MAGIC * **Secure** access and permission for model registrations, execution or modifications
# MAGIC
# MAGIC ## **What is this notebook doing?**
# MAGIC
# MAGIC We will look at how we test and promote a new __Challenger__ model as a candidate to replace an existing __Champion__ model.
# MAGIC
# MAGIC 1. Find best run and push model to UC
# MAGIC 2. Set model version alias
# MAGIC 3. Perform model checks (e.g. metrics)
# MAGIC 4. Validate results
# MAGIC 5. Promote validated model as "Champion"
# MAGIC
# MAGIC <img src="https://github.com/naradbx-dsa-blaze/Gilead-Sciences-DS-and-DE-Workshop/blob/feature/ddavis_model_registry/model-registry/img/ml-lifecycle-demo.png?raw=true" width="1000">
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://github.com/naradbx-dsa-blaze/Gilead-Sciences-DS-and-DE-Workshop/blob/feature/ddavis_model_registry/model-registry/img/ml-lifecycle-demo.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to use Models in Unity Catalog
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters. During the course of this development cycle, they will select the best run within an experiment and register its model to Unity Catalog.  Think of this as **committing** the model to the Unity Catalog, much as you would commit code to a version control system.
# MAGIC
# MAGIC Unity Catalog proposes free-text model alias i.e. `Baseline`, `Challenger`, `Champion` along with tagging.
# MAGIC
# MAGIC Users with appropriate permissions can create models, modify aliases and tags, use models etc.

# COMMAND ----------

# DBTITLE 1,Install MLflow version for model lineage in UC [for MLR < 15.2]
# MAGIC %pip install --quiet mlflow==2.14.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find best run and push model to the Unity Catalog for validation
# MAGIC
# MAGIC We have completed the training runs to find a candidate __Challenger__ model. We'll programatically select the best model from our last ML experiment and register it to Unity Catalog. We can easily do that using MLFlow `search_runs` API:

# COMMAND ----------

import mlflow

catalog = 'nara_catalog'
dbName = 'gilead_ds_workshop'
churn_experiment_name = "lr_influenza_outbreak"
model_name = f"{catalog}.{dbName}.dd_influenza_outbreak"
print(f"Finding best run from {churn_experiment_name}_* and pushing new model version to {model_name}")

xp_path = "/Shared/dbdemos/experiments/mlops"
experiment_id = mlflow.search_experiments(filter_string=f"name LIKE '{xp_path}%'", order_by=["last_update_time DESC"])[0].experiment_id
print(experiment_id)

# COMMAND ----------

# Let's get our best ml run
best_model = mlflow.search_runs(
  experiment_ids=experiment_id,
  order_by=["metrics.test_f1_score DESC"],
  max_results=1,
  filter_string="status = 'FINISHED' and run_name='mlops_best_run'" #filter on mlops_best_run to always use the notebook 02 to have a more predictable demo
)
# Optional: Load MLflow Experiment as a spark df and see all runs
# df = spark.read.format("mlflow-experiment").load(experiment_id)
best_model

# COMMAND ----------

# MAGIC %md Once we have our best model, we can now register it to the Unity Catalog Model Registry using it's run ID

# COMMAND ----------

print(f"Registering model to {model_name}")  # {model_name} is defined in the setup script

# Get the run id from the best model
run_id = best_model.iloc[0]['run_id']

# Register best model from experiments run to MLflow model registry
model_details = mlflow.register_model(f"runs:/{run_id}/sklearn_model", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC At this point the model does not yet have any aliases or description that indicates its lifecycle and meta-data/info.  Let's update this information.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Give the registered model a description
# MAGIC
# MAGIC We'll do this for the registered model overall.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# The main model description, typically done once.
client.update_registered_model(
  name=model_details.name,
  description="This model predicts whether a customer will churn using the features in the mlops_churn_training table. It is used to power the Telco Churn Dashboard in DB SQL.",
)

# COMMAND ----------

# MAGIC %md
# MAGIC And add some more details on the new version we just registered

# COMMAND ----------

# Provide more details on this specific model version
best_score = best_model['metrics.test_f1_score'].values[0]
run_name = best_model['tags.mlflow.runName'].values[0]
version_desc = f"This model version has an F1 validation metric of {round(best_score,4)*100}%. Follow the link to its training run for more details."

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description=version_desc
)

# We can also tag the model version with the F1 score for visibility
client.set_model_version_tag(
  name=model_details.name,
  version=model_details.version,
  key="f1_score",
  value=f"{round(best_score,4)}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the latest model version as the Challenger model
# MAGIC
# MAGIC We will set this newly registered model version as the __Challenger__ model. Challenger models are candidate models to replace the Champion model, which is the model currently in use.
# MAGIC
# MAGIC We will use the model's alias to indicate the stage it is at in its lifecycle.

# COMMAND ----------

# Set this version as the Challenger model, using its model alias
client.set_registered_model_alias(
  name=model_name,
  alias="Challenger",
  version=model_details.version
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now, visually inspect the model verions in Unity Catalog Explorer. You should see the version description and `Challenger` alias applied to the version.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Model information
# MAGIC
# MAGIC We will fetch the model information for the __Challenger__ model from Unity Catalog.

# COMMAND ----------

catalog = 'nara_catalog'
dbName = 'gilead_ds_workshop'

# We are interested in validating the Challenger model
model_alias = "Challenger"
model_name = f"{catalog}.{dbName}.mlops_churn"

client = MlflowClient()
model_details = client.get_model_version_by_alias(model_name, model_alias)
model_version = int(model_details.version)

print(f"Validating {model_alias} model for {model_name} on model version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model checks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Benchmark or business metrics on the eval dataset
# MAGIC
# MAGIC Let's use our validation dataset to check the potential new model impact.
# MAGIC
# MAGIC ***Note: This is just to evaluate our models, not to be confused with A/B testing**. A/B testing is done online, splitting the traffic to 2 models and requires a feedback loop to evaluate the effect of the prediction (e.g. after a prediction, did the discount we offered to the customer prevent the churn?). We will cover A/B testing in the advanced part.*

# COMMAND ----------

model_run_id = model_details.run_id
f1_score = mlflow.get_run(model_run_id).data.metrics['test_f1_score']

try:
    #Compare the challenger f1 score to the existing champion if it exists
    champion_model = client.get_model_version_by_alias(model_name, "Champion")
    champion_f1 = mlflow.get_run(champion_model.run_id).data.metrics['test_f1_score']
    print(f'Champion f1 score: {champion_f1}. Challenger f1 score: {f1_score}.')
    metric_f1_passed = f1_score >= champion_f1
except:
    print(f"No Champion found. Accept the model as it's the first one.")
    metric_f1_passed = True

print(f'Model {model_name} version {model_details.version} metric_f1_passed: {metric_f1_passed}')
# Tag that F1 metric check has passed
client.set_model_version_tag(name=model_name, version=model_details.version, key="metric_f1_passed", value=metric_f1_passed)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model performance metric
# MAGIC
# MAGIC We want to validate the model performance metric. Typically, we want to compare this metric obtained for the Challenger model agaist that of the Champion model. Since we have yet to register a Champion model, we will only retrieve the metric for the Challenger model without doing a comparison.
# MAGIC
# MAGIC The registered model captures information about the MLflow experiment run, where the model metrics were logged during training. This gives you traceability from the deployed model back to the initial training runs.
# MAGIC
# MAGIC Here, we will use the F1 score for the out-of-sample test data that was set aside at training time.

# COMMAND ----------

import pyspark.sql.functions as F
#get our validation dataset:
validation_df = spark.table('mlops_churn_training').filter("split='validate'")

#Call the model with the given alias and return the prediction
def predict_churn(validation_df, model_alias):
    model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{catalog}.{dbName}.mlops_churn@{model_alias}")
    return validation_df.withColumn('predictions', model(*model.metadata.get_input_schema().input_names()))

# COMMAND ----------

import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix

#Note: this is over-simplified and depends of your use-case, but the idea is to evaluate our model against business metrics
cost_of_customer_churn = 2000 #in dollar
cost_of_discount = 500 #in dollar

cost_true_negative = 0 #did not churn, we did not give him the discount
cost_false_negative = cost_of_customer_churn #did churn, we lost the customer
cost_true_positive = cost_of_customer_churn -cost_of_discount #We avoided churn with the discount
cost_false_positive = -cost_of_discount #doesn't churn, we gave the discount for free

def get_model_value_in_dollar(model_alias):
    print(model_alias)
    # Convert preds_df to Pandas DataFrame
    model_predictions = predict_churn(validation_df, model_alias).toPandas()
    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(model_predictions['churn'], model_predictions['predictions']).ravel()
    return tn * cost_true_negative+ fp * cost_false_positive + fn * cost_false_negative + tp * cost_true_positive

champion_potential_revenue_gain = get_model_value_in_dollar("Champion")
challenger_potential_revenue_gain = get_model_value_in_dollar("Challenger")

data = {'Model Alias': ['Challenger', 'Champion'],
        'Potential Revenue Gain': [challenger_potential_revenue_gain, champion_potential_revenue_gain]}

# Create a bar plot using plotly express
px.bar(data, x='Model Alias', y='Potential Revenue Gain', color='Model Alias',
       labels={'Potential Revenue Gain': 'Revenue Impacted'},
       title='Business Metrics - Revenue Impacted')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation results
# MAGIC
# MAGIC That's it! We have demonstrated some simple checks on the model. Let's take a look at the validation results.

# COMMAND ----------

results = client.get_model_version(model_name, model_version)
results.tags

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promoting the Challenger to Champion
# MAGIC
# MAGIC When we are satisfied with the results of the __Challenger__ model, we can then promote it to Champion. This is done by setting its alias to `@Champion`. Inference pipelines that load the model using the `@Champion` alias will then be loading this new model. The alias on the older Champion model, if there is one, will be automatically unset. The model retains its `@Challenger` alias until a newer Challenger model is deployed with the alias to replace it.

# COMMAND ----------

if  results.tags["metric_f1_passed"]:
  print('register model as Champion!')
  client.set_registered_model_alias(
    name=model_name,
    alias="Champion",
    version=model_version
  )
else:
  raise Exception("Model not ready for promotion")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Congratulations, our model is now validated and promoted accordingly
# MAGIC
# MAGIC We now have the certainty that our model is ready to be used in inference pipelines and in realtime serving endpoints, as it matches our validation standards.
# MAGIC
# MAGIC
# MAGIC Next: [Run batch inference from our newly promoted Champion model]($./05_batch_inference)
