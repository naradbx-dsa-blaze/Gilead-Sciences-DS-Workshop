-- Databricks notebook source
-- MAGIC %md 
-- MAGIC ## Persist DLT streaming view
-- MAGIC To easily support DLT / UC / ML during the preview with all cluster types, we temporary recopy the final DLT view to another UC table 

-- COMMAND ----------

CREATE OR REPLACE TABLE nara_catalog.gilead_de_workshop.drug_exposure_ml AS SELECT * FROM nara_catalog.gilead_de_workshop.drug_exposure;
CREATE OR REPLACE TABLE nara_catalog.gilead_de_workshop.person_ml AS SELECT * FROM nara_catalog.gilead_de_workshop.person;
CREATE OR REPLACE TABLE nara_catalog.gilead_de_workshop.patients_ml AS SELECT * FROM nara_catalog.gilead_de_workshop.patients;
CREATE OR REPLACE TABLE nara_catalog.gilead_de_workshop.encounters_ml AS SELECT * FROM nara_catalog.gilead_de_workshop.encounters;
CREATE OR REPLACE TABLE nara_catalog.gilead_de_workshop.condition_occurrence_ml AS SELECT * FROM nara_catalog.gilead_de_workshop.condition_occurrence;
CREATE OR REPLACE TABLE nara_catalog.gilead_de_workshop.conditions_ml AS SELECT * FROM nara_catalog.gilead_de_workshop.conditions;
