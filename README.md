MRI based dementia prediction

This repository uses the kaggle dataset "MRI and Alzheimer" from https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers/data. This is my capstone project of the MachineLearningOperations course "MLOps Zoomcamp" from DataTalksClubs (https://github.com/DataTalksClub/mlops-zoomcamp).  

Background:  
Early diagnosis of dementia, of which Alzheimer is one specific form, is crucial as irreversible damage already occurs before the onset of symptoms. The data used in this repository are collected with Magnet Resonance Imaging (MRI), which allows imaging inner body structures, such as the brain. The measured parameters are then correlated with clinical tests for dementia for demented and non-demented individuals.  

Based on this data I use machine learning to be able to predict the probability of dementia based solely on MRI scans, to predict individuals with high-risk of dementia before the onset of symptoms and hence a potential treatment in an early phase of the disease. I thus use the brain parameters measured with MRI to classify patients as demented and non-demented. In practice, the demented patients would then retrieve treatment as early as possible to prevent a further spread of the dementia.  

Dataset:  
There are two datasets available: One dataset in which each patient was scanned once (cross sectional dataset), and another independent dataset in which each patient was scanned multiple times over several years (longitudinal dataset). There were two possible ways of dealing with the dataset: either merging them, but unsing only the first scan of each patient in the longitudinal dataset (as otherwise the measurements wouldn't be independent any more), or focusing on the longitudinal datasets to see if one can predict onset of dementia at a later stage. Predicting onset of dementia based on the longitudinal dataset performed very poorly on initial trials, hence I merged both datasets and discarded additional measurements of the same patient.  

The combined dataset contains 15 columns [  
'id': patient id,  
'm/f':gender,  
'hand': preferred hand,  
'age': age in years,  
'educ': education level on a scale from 1 (minimal) to 5,  
'ses': socioeconomic status on a scale from 1 to 5,  
'mmse': mental state exam - measurement of cognitive performance on a scale from 30 to 0 - scores <=26 mean dementia,  
'cdr': cognitive dementia ranking - measurement of cognitive performance on a scale from 0 to 3 in which a score >=1 means dementia,  
'etiv': estimated volume of the skull,  
'nwbv': normalized volume of the brain,  
'asf': atlas scaling factor used for brain volume normalization,  
'delay': time elapsed in between visits (only longitudinal dataset),  
'mri_id': scan id (only longitudinal dataset),  
'group': classification as demented or not prior to MRI scan (only longitudinal dataset),  
'visit': number of visit (only longitudinal dataset].  

Notably, only two of these parameters are relevant readouts from the scans and hence 'hard' predictors for dementia: etiv and nwbv. The factors gender, age, educ and ses can be seen as 'soft' predictors, which in combination with the MRI scans can predict dementia. hand (identic across all patients) and id have no informative value. mmse and cdr are both readouts of cognitive performance (which are independent of the MRI scan) and could thus be used as target variable to define patients as demented/non-demented. No matter which of these is chosen, it is crucial to remove the other one from the training dataset.   

The potentially relevant columns for the outlined purpose are thus:  

'm/f':gender,  
'hand': preferred hand,  
'age': age in years,  
'educ': education level on a scale from 1 (minimal) to 5,  
'ses': socioeconomic status on a scale from 1 to 5,  
'etiv': estimated volume of the skull,  
'nwbv': normalized volume of the brain,  
'asf': atlas scaling factor used for brain volume normalization.   

The model is orchestrated with prefect - code blocks are assigned with the @flow and @task decorators and the model is deployed to the prefect cloud.

The target variable used is the ratio of:  
'mmse': mental state exam - measurement of cognitive performance on a scale from 30 to 0 - scores <=26 mean dementia,  
'cdr': cognitive dementia ranking - measurement of cognitive performance on a scale from 0 to 3 in which a score >=1 means dementia. 
As a mmse score <=26 means dementia and a cdr score >=1 means dementia, a ratio of cdr/mmse >= 1/26 would mean dementia, which I use as threshold for binary classification. 

ML runs are tracked with mlflow and the workflow is orchestrated using prefect. All data is stored on github codespaces. The model is hosted as a webservice using gunicorn and can be downloaded as docker image. Relevant commands that need to be entered in the terminal for reproducing the code can be found in the commands file. Unit tests can be found in the test folder.
