"""
Utility functions for logging data to mlflow and to load it

"""

from mlflow.data import from_pandas
import tempfile
import os
import mlflow
import pandas as pd

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
        mlflow.log_artifact(local_path = file_path, artifact_path="datasets/" + name)

def load_logged_dataset(run_id, dataset_path):
    """
    Load dataset from mlflow

    Args: 
        run_id (str): run id. The run id is the id of the run that logged the dataset
        dataset_path (str): path to dataset. The path is relative to the mlruns folder    
    """
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=dataset_path)
    return pd.read_parquet(local_path)