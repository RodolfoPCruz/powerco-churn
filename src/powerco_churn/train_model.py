from pipeline.build_pipeline import get_preprocess_pipeline
from pathlib import Path
import pandas as pd
import mlflow
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
import argparse

current_file = Path(__file__).resolve()
base_path = current_file.parents[2]  # 0 is file, 1 is parent, ..., 3 = three levels up
base_path = str(base_path)

client_data_train = pd.read_csv(base_path + '/data/raw/train/train_client_data_raw.csv')
price_data_train = pd.read_csv(base_path + '/data/raw/train/train_price_data_raw.csv')
y_train = pd.read_csv(base_path + '/data/raw/train/y_train.csv')

client_data_test = pd.read_csv(base_path + '/data/raw/test/test_client_data_raw.csv')
price_data_test  = pd.read_csv(base_path + '/data/raw/test/test_price_data_raw.csv')
y_test = pd.read_csv(base_path + '/data/raw/test/y_test.csv')    


mlflow.set_tracking_uri(f"file:{base_path}/mlruns")  # Goes one level up
mlflow.set_experiment("powerco_churn")

with mlflow.start_run(run_name="baseline_lightgbm"):

    
    model_lgbm = LGBMClassifier( class_weight = 'balanced', 
                                 scale_pos_weight = 9.0, 
                                n_estimators = 100,
                                random_state = 42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall'
    }

    pre_process_pipeline = get_preprocess_pipeline()
    x_train = pre_process_pipeline.fit_transform([client_data_train, price_data_train])
    x_test = pre_process_pipeline.transform([client_data_train, price_data_train])

    results = cross_validate(model_lgbm, 
                             x_train, 
                             y_train,
                            cv = 5, 
                            scoring=scoring)
    results_df = pd.DataFrame(results)
    results_df.to_csv('artifacts/cross_val_metrics.csv')
    mlflow.log_artifact("artifacts/cross_val_metrics.csv")

    print("Accuracy: ", results['test_accuracy'].mean())
    print("Accuracy per fold:", results['test_accuracy'])
    print("Precision per fold:", results['test_precision'])
    print("Recall per fold:", results['test_recall'])

    print(len(x_train))
    print(len(x_test))
    print(len(y_train))
    print(len(y_test))

    '''
    model_lgbm.fit(x_train, y_train)
    y_pred = model_lgbm.predict(x_test)
     
    acc_test = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", acc_test)

    precision_test = precision_score(y_test, y_pred)
    mlflow.log_metric("test_precision", precision_test)

    recall_test = recall_score(y_test, y_pred)
    mlflow.log_metric("test_recall", recall_test)

    mlflow.sklearn.log_model(model_lgbm, "model")
    '''

    

'''
if __name__ == '__main__':
    pipeline = get_pipeline()
''' 