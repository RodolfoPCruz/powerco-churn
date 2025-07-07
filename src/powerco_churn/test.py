from pipeline.build_pipeline import get_preprocess_pipeline
from pathlib import Path
import pandas as pd
import mlflow
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
import argparse
from powerco_churn.utils.log_mlflow_utils import load_logged_dataset

current_file = Path(__file__).resolve()
base_path = current_file.parents[2]  # 0 is file, 1 is parent, ..., 3 = three levels up
base_path = str(base_path)

run_id = '37a11a4d37394a01a54f562039a5ba83'

mlflow.set_tracking_uri(f"file:{base_path}/mlruns")  # Goes one level up
mlflow.set_experiment("powerco_churn")

client_data_train = load_logged_dataset(run_id, 'datasets/client_train_data/client_train_data.parquet')
client_data_test = load_logged_dataset(run_id, 'datasets/client_test_data/client_test_data.parquet')

print(client_data_train.shape)

