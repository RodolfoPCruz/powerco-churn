from pipeline.build_pipeline import get_preprocess_pipeline
from pathlib import Path
import pandas as pd
import mlflow
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
import argparse
from powerco_churn.utils.log_mlflow_utils import load_logged_dataset
from sklearn.pipeline import Pipeline

current_file = Path(__file__).resolve()
base_path = current_file.parents[2]  # 0 is file, 1 is parent, ..., 3 = three levels up
base_path = str(base_path)

#run id generated when the raw dataset was logged
run_id_raw_dataset = '8cd45ac031db4540b94bdc7efd917411'

mlflow.set_tracking_uri(f"file:{base_path}/mlruns")  # Goes one level up
mlflow.set_experiment("powerco_churn")

#loading raw dataset
client_data_train = load_logged_dataset(run_id_raw_dataset, 'datasets/client_train_data/client_train_data.parquet')
client_data_test = load_logged_dataset(run_id_raw_dataset, 'datasets/client_test_data/client_test_data.parquet')

y_train = load_logged_dataset(run_id_raw_dataset, 'datasets/y_train/y_train.parquet')
y_test = load_logged_dataset(run_id_raw_dataset, 'datasets/y_test/y_test.parquet')

price_data_train = load_logged_dataset(run_id_raw_dataset, 'datasets/price_train_data/price_train_data.parquet')
price_data_test = load_logged_dataset(run_id_raw_dataset, 'datasets/price_test_data/price_test_data.parquet')



parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type = int, default = 100, help =" Number of trees in the forest")
parser.add_argument("--scale_pos_weight", type = float, default = 9.0, help = "Weights associated with classes ")
parser.add_argument("--random_state", type = int, default = 42, help = "Random number seed")
parser.add_argument("--class_weight", type = str, default = 'balanced', help = "Weights associated with classes")

args = parser.parse_args()

model_lgbm = LGBMClassifier(class_weight = args.class_weight,
                            scale_pos_weight = args.scale_pos_weight, 
                            n_estimators = args.n_estimators,
                            random_state = args.random_state)
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall'
}

#run id generated when the pre-processing pipeline was logged
run_id_pipeline = '8142c0c2ca004887a918401cf94bf952'
pre_process_pipeline = mlflow.sklearn.load_model(f"runs:/{run_id_pipeline}/preprocessing_pipeline")

#preprocessing the raw data
x_train = pre_process_pipeline.fit_transform([client_data_train, price_data_train])
x_test = pre_process_pipeline.transform([client_data_test, price_data_test])

#training the model using cross validation
results = cross_validate(model_lgbm, 
                            x_train, 
                            y_train,
                        cv = 5, 
                        scoring=scoring)
results_df = pd.DataFrame(results)
results_df.to_csv('src/powerco_churn/artifacts/cross_val_metrics.csv')
#mlflow.log_artifact("src/powerco_churn/artifacts/cross_val_metrics.csv")

#training the model using the enrire dataset. The model will be evaluated
#using the test data
model_lgbm.fit(x_train, y_train)
y_pred = model_lgbm.predict(x_test)
    
acc_test = accuracy_score(y_test, y_pred)
#mlflow.log_metric("test_accuracy", acc_test)

precision_test = precision_score(y_test, y_pred)
#mlflow.log_metric("test_precision", precision_test)

recall_test = recall_score(y_test, y_pred)
#mlflow.log_metric("test_recall", recall_test)


#Creating a new pipeline that integrates the preprocessing step and the model





#mlflow.sklearn.log_model(model_lgbm, "model")
with mlflow.start_run(run_name = "train_model"):
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("scale_pos_weight", args.scale_pos_weight)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("class_weight", args.class_weight)
    mlflow.log_artifact("src/powerco_churn/artifacts/cross_val_metrics.csv")
    mlflow.log_metric("test_accuracy", acc_test)
    mlflow.log_metric("test_precision", precision_test)
    mlflow.log_metric("test_recall", recall_test)
    mlflow.sklearn.log_model(model_lgbm, "model")
