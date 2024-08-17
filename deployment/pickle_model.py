import pickle
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

model_name = "nyc-taxi-regressor"
latest_versions = client.get_latest_versions(name=model_name)


with open('./output/train.pkl', 'wb') as f_out:
   pickle.dump((X_train,y_train), f_out)
f_out.close()
    