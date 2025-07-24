import mlflow.sklearn
import pandas as pd
import mlflow
import os
from powerco_churn.utils.logger_utils import configure_logging
from mlflow.artifacts import download_artifacts
import logging
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score


import sys
print("\n".join(sys.path))

raw_dataset_run_id = '8cd45ac031db4540b94bdc7efd917411'
model_run_id = '31385a7783574529b30b8b7cf2d96e67'

client_data_artifact_test = "datasets/client_test_data/client_test_data.parquet"
local_path_client_data_test = "'data/raw/test/test_client_data_raw.csv'"

price_data_artifact_test = "datasets/price_test_data/price_test_data.parquet"
local_path_price_data_test = 'data/raw/test/test_price_data_raw.csv'

y_test_artifact = "datasets/y_test/y_test.parquet" 
local_path_y_test = 'data/raw/test/y_test.csv'

configure_logging(log_file_name = "main.log")

def load_data(run_id, artifact_path, local_path):
    """
    Try to load the dataset 

    Args: 
        run_id (str): run id. The run id is the id of the run that logged the dataset
        dataset_path (str): path to dataset. The path is relative to the mlruns folder    
    """
    file_name = artifact_path.split("/")[1]
    logging.info(f"Loading {file_name}")

    try:

        path = mlflow.artifacts.download_artifacts(run_id = run_id,
                                                    artifact_path = artifact_path)
        return pd.read_parquet(path)
        
    except Exception as e:
        logging.info(f"Could not download from mlflow: {e}")
        logging.info(f"Falling back to local file: {local_path}")
        if not os.path.exists(local_path):
            logging.info(f"Could not find local file: {local_path}")
            raise FileNotFoundError(f"Could not find local file: {local_path}")
        return  pd.read_csv(local_path)
    

def load_model(run_id = model_run_id):
    """
    Load model from mlflow
    """

    logging.info('Logging model from mlflow')
    model = mlflow.sklearn.load_model(f"runs:/{model_run_id}/complete_pipeline")

    return model

def main():

    logging.info("Starting inference pipeline")



    client_test_data = load_data(run_id = raw_dataset_run_id, 
                                artifact_path = client_data_artifact_test,
                                local_path = local_path_client_data_test)


    price_test_data = load_data(run_id = raw_dataset_run_id, 
                                artifact_path = price_data_artifact_test,
                                local_path = local_path_price_data_test)


    y_test = load_data(run_id = raw_dataset_run_id, 
                                    artifact_path = y_test_artifact,
                                    local_path = local_path_y_test)    
   
    complete_pipeline = mlflow.sklearn.load_model(f"runs:/{'31385a7783574529b30b8b7cf2d96e67'}/complete_pipeline")

    X_input = [client_test_data, price_test_data]
    y_pred = complete_pipeline.predict(X_input)
    
    acc_test = accuracy_score(y_test, y_pred)
    #mlflow.log_metric("test_accuracy", acc_test)

    precision_test = precision_score(y_test, y_pred)
    #mlflow.log_metric("test_precision", precision_test)

    recall_test = recall_score(y_test, y_pred)
    #mlflow.log_metric("test_recall", recall_test)

    mlflow.set_tracking_uri(f"file:mlruns")  # Goes one level up
    mlflow.set_experiment("powerco_churn")

    with mlflow.start_run(run_name = "inference_pipeline"):
        mlflow.log_metric("test_accuracy", acc_test)
        mlflow.log_metric("test_precision", precision_test)
        mlflow.log_metric("test_recall", recall_test)


if __name__ == "__main__":
    main()