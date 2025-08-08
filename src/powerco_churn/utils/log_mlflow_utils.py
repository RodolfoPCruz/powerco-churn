"""
Utility functions for logging data to mlflow and to load it

"""

from mlflow.data import from_pandas
from mlflow.tracking import MlflowClient
import tempfile
import os
import mlflow
import pandas as pd
import logging

def log_dataset_mlflow(df, source, name, context = "training"):
    """
    Log dataset to mlflow

    Args:
        df (pd.DataFrame): dataframe to log
        source (str): source of dataframe. It's metadata — not a path used to load or store the data
        name (str): name of dataframe
        context (str, optional): context of dataframe. Defaults to "training".
    """
    dataset = from_pandas(df, source = source, name = name)
    mlflow.log_input(dataset, context = context)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, name + ".parquet")
        df.to_parquet(file_path)
        mlflow.log_artifact(local_path = file_path, artifact_path = "datasets/" + name)

def load_logged_dataset(run_id, artifact_path, local_path):
    """
    Try MLflow first; if it fails, manually resolve the artifact path.

    Args: 
        run_id (str): run id. The run id is the id of the run that logged the dataset
        artifact_path (str): path to the dataset in mlflow
    """
    tracking_uri = mlflow.get_tracking_uri().replace("file:", "")
    #print(f"Tracking URI: {tracking_uri}")
    
    #verify wether the run_id exists
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        logging.info("✅ Run exists!")
    except Exception as e:
        if os.path.exists(local_path):
            logging.info("✅Run id does not exist. Loading from local file")
            return pd.read_csv(local_path)
        else:
            raise FileNotFoundError(f"The mlflow run does not exist and it was not possible to find a local file at {local_path}")

    
    
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        logging.info('Loading dataset from mlflow')
        return pd.read_parquet(local_path)
    
    except Exception as e:
        print(f"MLflow failed: {e}")
        print("Falling back to manual resolution...")

        #Manual artifact resolution
        experiment_folders = os.listdir(tracking_uri)
        experiment_id = next(
            folder for folder in experiment_folders
            if os.path.isdir(os.path.join(tracking_uri, folder, run_id))
        )
        manual_path = os.path.join(tracking_uri, experiment_id, run_id, 
                                   "artifacts", artifact_path)
        logging.info("Manual resolved path:", manual_path)
        if os.path.exists(manual_path):
            return pd.read_parquet(manual_path)
        
        if os.path.exists(local_path):
            return pd.read_clipboard(local_path)
        
        raise FileNotFoundError(f"Could not find artifact {artifact_path} "
                                "in MLflow or locally.")
