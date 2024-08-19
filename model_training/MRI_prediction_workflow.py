import pandas as pd
import numpy as np
import os
import mlflow

from prefect import flow, task
from prefect.tasks import task_input_hash
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor
from datetime import timedelta

@task(retries=3, retry_delay_seconds=10, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def get_full_path(filename: str) -> str:
    """
    Get the full path of the file based on the current working directory.
    
    Args:
        filename (str): Name of the file.
    
    Returns:
        str: Full path to the file.
    """
    cwd = os.getcwd()
    return cwd + filename

@task(retries=3, retry_delay_seconds=10, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_data(full_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.
    
    Args:
        full_path (str): Full path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(full_path, sep=",", engine="python", on_bad_lines="skip")
    return df

@task(retries=3, retry_delay_seconds=10)
def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the column names of the DataFrame by replacing spaces with underscores and converting to lowercase.
    
    Args:
        df (pd.DataFrame): DataFrame with original column names.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

@task(retries=3, retry_delay_seconds=10)
def merge_data(df_cross: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cross-sectional and longitudinal data into a single DataFrame.
    
    Args:
        df_cross (pd.DataFrame): Cross-sectional data.
        df_long (pd.DataFrame): Longitudinal data.
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    new_column_names = {
        'subject_id': 'id',
        'mr_delay': 'delay'
    }

    df_long.rename(columns=new_column_names, inplace=True)

    new_column_order = ['id', 'm/f', 'hand', 'age', 'educ', 'ses', 'mmse', 'cdr', 'etiv',
                        'nwbv', 'asf', 'delay', 'mri_id', 'group', 'visit']

    df_long = df_long[new_column_order]

    df = pd.merge(df_cross, df_long, on=['id', 'm/f', 'hand', 'age', 'educ', 'ses', 'mmse', 'cdr', 'etiv',
                                         'nwbv', 'asf', 'delay'], how='outer')
    return df

@task(retries=3, retry_delay_seconds=10)
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by dropping unnecessary columns and rows with missing information.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df[df['visit'].isna() | (df['visit'] == 1)]
    df_clean = df_clean.drop(columns=['hand', 'delay', 'id', 'mri_id', 'group', 'visit', 'asf'])
    df_clean = df_clean.dropna(subset=['cdr'])
    df_clean = df_clean.dropna(subset=['mmse'])
    df_clean['m/f'] = (df_clean['m/f'] == "M").astype(int)
    df_clean = df_clean.fillna(0)
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

@task(retries=3, retry_delay_seconds=10)
def create_binary_fusion_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target for dementia prediction based on CDR and MMSE scores.
    
    Args:
        df (pd.DataFrame): DataFrame with CDR and MMSE scores.
    
    Returns:
        pd.DataFrame: DataFrame with binary target.
    """
    df['target'] = np.log1p((df['cdr'] + 0.5) / df['mmse'])
    dementia_threshold = np.log1p(1.5 / 26)
    df['target'] = (df['target'] > dementia_threshold).astype(int)
    return df

@task(retries=3, retry_delay_seconds=10)
def split_data(df: pd.DataFrame, seed: int):
    """
    Split the data into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        seed (int): Random seed for reproducibility.
    
    Returns:
        tuple: X_train, X_val, y_train, y_val DataFrames and Series for training and validation.
    """
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train.target
    y_train = df_train.target
    y_val = df_val.target
    y_test = df_test.target

    df_full_train = df_full_train.drop(columns=['target', 'mmse', 'cdr'])
    df_train = df_train.drop(columns=['target', 'mmse', 'cdr'])
    df_val = df_val.drop(columns=['target', 'mmse', 'cdr'])
    df_test = df_test.drop(columns=['target', 'mmse', 'cdr'])

    X_train = df_train
    X_val = df_val

    return X_train, X_val, y_train, y_val

@task(retries=3, retry_delay_seconds=10)
def get_scores(y_val, y_pred) -> tuple:
    """
    Calculate performance metrics for model evaluation.
    
    Args:
        y_val (pd.Series): True labels for the validation set.
        y_pred (np.ndarray): Predicted labels for the validation set.
    
    Returns:
        tuple: ROC AUC score, accuracy, precision, recall, and F1 score.
    """
    roc_auc = roc_auc_score(y_val, y_pred)
    accuracy = sum(y_val == y_pred) / len(y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return roc_auc, accuracy, precision, recall, f1

@task(retries=3, retry_delay_seconds=10, timeout_seconds=3600)
def explore_models(X_train, y_train, X_val, y_val, num_trials: int, seed, model_type: str):
    """
    Optimize a machine learning model using Hyperopt.
    
    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data labels.
        X_val (pd.DataFrame): Validation data features.
        y_val (pd.Series): Validation data labels.
        num_trials (int): Number of trials for optimization.
        seed (int): Random seed for reproducibility.
        model_type (str): Type of model to optimize ('random_forest', 'logistic_regression', 'xgboost').
    
    Returns:
        None
    """
    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)

            if model_type == "random_forest":
                model = RandomForestRegressor(**params)
            elif model_type == "logistic_regression":
                model = LogisticRegression(**params)
            elif model_type == "xgboost":
                model = XGBRegressor(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if model_type in ["random_forest", "logistic_regression", "xgboost"]:
                dementia_threshold = np.log1p(1.5 / 26)
                y_pred = (y_pred > dementia_threshold).astype(int)

            roc_auc, accuracy, precision, recall, f1 = get_scores(y_val, y_pred)

            mlflow.log_metric("ROC_AUC_score", roc_auc)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.sklearn.log_model(model, artifact_path="models_mlflow")
            mlflow.set_tag("model", model)
            # artifact_uri = mlflow.get_artifact_uri()
            # mlflow.log_param("artifact_uri", artifact_uri)
            mlflow.end_run()

        return {'loss': -f1, 'status': STATUS_OK}

    if model_type == "random_forest":
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
            'random_state': seed
        }
    elif model_type == "logistic_regression":
        search_space = {
            'C': scope.float(hp.quniform('C', 0.001, 10, 0.5)),
            'solver': hp.choice('solver', ['lbfgs', 'newton-cg', 'liblinear']),
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': seed
        }
    elif model_type == "xgboost":
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 50)),
            'subsample': hp.uniform('subsample', 0.7, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
            'random_state': seed
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    rstate = np.random.default_rng(seed)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

@flow
def main_flow(
        cross_file: str = '/data/oasis_cross-sectional.csv',
        long_file: str = '/data/oasis_longitudinal.csv',
        seed=42,
        num_trials=15
):
    """
    Main flow orchestrating the entire MRI prediction workflow.

    Args:
        cross_file (str): Path to the cross-sectional data file.
        long_file (str): Path to the longitudinal data file.
        seed (int): Random seed for reproducibility.
        num_trials (int): Number of trials for model optimization.

    Returns:
        None
    """
    # mlflow setup
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("C")
    
    # load data
    df_cross = clean_col_names(load_data(get_full_path(cross_file)))
    df_long = clean_col_names(load_data(get_full_path(long_file)))

    # transform data
    df = merge_data(df_cross, df_long)
    df = clean_df(df)
    df = create_binary_fusion_target(df)
    X_train, X_val, y_train, y_val = split_data(df, seed)

    # train models
    model_types = ["random_forest", "logistic_regression", "xgboost"]

    for model_type in model_types:
        print(f"Optimizing model: {model_type}")
        explore_models(X_train, y_train, X_val, y_val, num_trials, seed, model_type)
        print(f"Finished optimizing model: {model_type}\n")

if __name__ == "__main__":
    main_flow()
