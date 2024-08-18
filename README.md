# MRI-Based Dementia Prediction

This repository utilizes the Kaggle dataset ["MRI and Alzheimer's"](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers/data) for a capstone project in the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course by DataTalksClub.

## Background

Early diagnosis of dementia, including Alzheimer's disease, is crucial as irreversible damage can occur before the onset of symptoms. This repository uses MRI data to correlate brain imaging parameters with clinical tests for dementia in demented and non-demented individuals.

The goal is to predict the probability of dementia based solely on MRI scans. This can help identify individuals at high risk of dementia before symptoms appear, allowing for potential early treatment. The brain parameters measured with MRI are used to classify patients as demented or non-demented. In practice, early identification could lead to timely interventions to prevent the further spread of dementia.

## Dataset

Two datasets are available:

1. **Cross-sectional dataset**: Each patient was scanned once.
2. **Longitudinal dataset**: Each patient was scanned multiple times over several years.

Predicting onset of dementia based on the longitudinal dataset performed very poorly on initial trials, hence I merged both datasets and discarded additional measurements of the same patient to avoid dependency issues. The combined dataset contains 15 columns:

- `id`: Patient ID
- `m/f`: Gender
- `hand`: Preferred hand
- `age`: Age in years
- `educ`: Education level (scale from 1 to 5)
- `ses`: Socioeconomic status (scale from 1 to 5)
- `mmse`: Mental state exam score (scale from 30 to 0, where scores ≤26 indicate dementia)
- `cdr`: Cognitive dementia ranking (scale from 0 to 3, where scores ≥1 indicate dementia)
- `etiv`: Estimated total intracranial volume
- `nwbv`: Normalized whole brain volume
- `asf`: Atlas scaling factor (used for brain volume normalization)
- `delay`: Time elapsed between visits (only in longitudinal dataset)
- `mri_id`: MRI scan ID (only in longitudinal dataset)
- `group`: Classification as demented or non-demented prior to MRI scan (only in longitudinal dataset)
- `visit`: Number of visits (only in longitudinal dataset)

### Relevant Predictors

Out of these, the following parameters are particularly relevant for predicting dementia:

- **Hard Predictors** (directly measured from MRI scans):
  - `etiv`: Estimated total intracranial volume
  - `nwbv`: Normalized whole brain volume
  
- **Soft Predictors** (demographic and socioeconomic factors):
  - `m/f`: Gender
  - `age`: Age
  - `educ`: Education level
  - `ses`: Socioeconomic status

The `hand` and `id` columns have no informative value. The `mmse` and `cdr` scores, which measure cognitive performance, can be used as target variables for defining whether patients are demented. It is crucial to remove one if the other is used as a target variable in training.

## Model Details

The model is orchestrated using Prefect, with code blocks decorated with `@flow` and `@task`. The model is deployed to Prefect Cloud.

### Target Variable

The target variable is the ratio of:

- `mmse`: Mental state exam score
- `cdr`: Cognitive dementia ranking

As on the mmse scale a score <=26 indicates dementia and on the cdr scale a score >=1 indicates dementia, a ratio of cdr/mmse >= 1/26 would indicate dementia on a merged scale. I used this as the threshold for binary classification. 

## Technical Implementation

- **ML Runs Tracking**: Managed with MLflow.
- **Orchestration**: Workflow orchestrated using Prefect.
- **Data Storage**: Hosted on GitHub Codespaces.
- **Model Hosting**: Deployed as a web service using Gunicorn, and available as a Docker image.

## Reproducibility

Relevant commands for reproducing the code can be found in the `commands` file. Unit tests are located in the `test` folder.
