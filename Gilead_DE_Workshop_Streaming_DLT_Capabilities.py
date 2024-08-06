# Databricks notebook source
# MAGIC %pip install dbdemos

# COMMAND ----------

import dbdemos
dbdemos.install('lakehouse-hls-readmission', catalog='nara_catalog', schema='gilead_de_workshop')

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
