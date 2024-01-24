"""The code loads a machine learning model and configuration settings from JSON, predicts results on an inference dataset, and logs the outcome."""
import json
import logging
import os
import pickle
import sys
from datetime import datetime

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from dotenv import load_dotenv

from utils import get_project_dir, configure_logging

load_dotenv()

# Adds the root directory to the system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

# Loads configuration settings from JSON
with open("../" + CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

INF_DATA = os.path.join(DATA_DIR, conf['inference']['inf_table_name'])


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pickle') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pickle'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path, input_size, output_size):
    """Loads and returns the specified model"""
    try:
        with open(path, 'rb') as f:
            # Load the class definition and instantiate the model
            model_class = pickle.load(f)

            model_instance = model_class(input_size, output_size)
            model_instance.load_state_dict(pickle.load(f))  # Load the model state dict
            logging.info(f'Path of the model: {path}')
            return model_instance
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """Loads and returns data for inference from the specified CSV file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model: DecisionTreeClassifier, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict the results and join them with the infer_data"""
    results = model.predict(infer_data)
    infer_data['results'] = results
    return infer_data


def store_results(results: pd.DataFrame) -> None:
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
    path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    infer_data = get_inference_data(os.path.join(DATA_DIR, INF_DATA))
    le = LabelEncoder()
    infer_data['Species'] = le.fit_transform(infer_data['Species'])
    X = torch.FloatTensor(infer_data.drop('Species', axis=1).values)
    y = torch.LongTensor(infer_data['Species'].values)

    input_size = X.shape[1]
    output_size = len(torch.unique(y))

    model = get_model_by_path(get_latest_model_path(), input_size, output_size)

    results = predict_results(model, infer_data)
    store_results(results)
    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()
