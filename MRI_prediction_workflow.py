#MRI_prediction_workflow
import pandas as pd
import numpy as np
import os 
import mlflow

from prefect import flow, task
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor

def get_full_path(filename: str):    
    cwd = os.getcwd()
    return cwd+filename


def load_data(full_path: str):
    """Load data into df"""
    
    df = pd.read_csv(full_path,sep=",", engine="python", on_bad_lines="skip")     
    return df

def clean_col_names(df):
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

def merge_data(df_cross, df_long):
    
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

def clean_df(df):
    """drop cols and rows without information"""
    df_clean = df[df['visit'].isna() | (df['visit'] == 1)]
    df_clean = df_clean.drop(columns=['hand', 'delay', 'id', 'mri_id','group','visit','asf'])
    df_clean = df_clean.dropna(subset=['cdr'])
    df_clean = df_clean.dropna(subset=['mmse'])
    df_clean['m/f'] = (df_clean['m/f'] == "M").astype(int)
    df_clean=df_clean.fillna(0)
    df_clean = df_clean.reset_index(drop=True)
    return df_clean
    
def create_binary_fusion_target(df):
    df['target']=np.log1p((df['cdr']+0.5)/df['mmse'])
    dementia_threshold = np.log1p(1.5/26)
    df['target'] = (df['target'] > dementia_threshold).astype(int)
    return df

def split_data(df, seed):
    
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

    df_full_train = df_full_train.drop(columns = ['target','mmse','cdr'])
    df_train = df_train.drop(columns = ['target','mmse','cdr'])
    df_val = df_val.drop(columns = ['target','mmse','cdr'])
    df_test = df_test.drop(columns = ['target','mmse','cdr'])

    X_train = df_train
    X_val = df_val

    return X_train, X_val, y_train, y_val


def get_scores(y_val, y_pred):
    roc_auc = roc_auc_score(y_val,y_pred)
    accuracy = sum(y_val==y_pred)/len(y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return roc_auc, accuracy, precision, recall, f1


def optimize_model(X_train, y_train, X_val, y_val, num_trials: int, seed, model_type: str):

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
            mlflow.sklearn.log_model(model, artifact_path="artifact")
            mlflow.set_tag("model", model)
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

def main_flow(
        cross_file: str = '/data/oasis_cross-sectional.csv',
        long_file: str = '/data/oasis_longitudinal.csv',
        seed = 42,
        num_trials = 15
):
        #mlflow setup
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("random-forest-hyperopt")

        #load
        df_cross = clean_col_names(load_data(get_full_path(cross_file)))
        df_long = clean_col_names(load_data(get_full_path(long_file)))

        #transform
        df = merge_data(df_cross, df_long)
        df = clean_df(df)
        df = create_binary_fusion_target(df)
        X_train, X_val, y_train, y_val = split_data(df, seed)
        
        #train
        model_types = ["random_forest", "logistic_regression", "xgboost"]
        for model_type in model_types:
            print(f"Optimizing model: {model_type}")
            optimize_model(X_train, y_train, X_val, y_val, num_trials, seed, model_type)
            print(f"Finished optimizing model: {model_type}\n")
        
if __name__ == "__main__":
    main_flow()



