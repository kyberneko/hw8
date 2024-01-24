"""The script creates and saves two datasets, one for training (TRAIN_PATH) and one for inference (INFERENCE_PATH),
using the IrisDataSetGenerator singleton class. The data is loaded from a Wikipedia link, and the script logs the process. """
from utils import singleton, get_project_dir, configure_logging
import pandas as pd
import logging
import os
import sys
import json
from dotenv import load_dotenv

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Get settings file from config
load_dotenv()  # load_dotenv() helps with accessing env file
CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open("../" + CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inf_table_name'])

# Singleton class for generating Iris data set
@singleton
class IrisDataSetGenerator():
    def __init__(self):
        self.df = None

    # Method to create the Iris dataset
    def create(self, save_path: str):
        logger.info("Creating Iris dataset...")
        self.df = self._load_data()
        self.save(self.df, save_path)
        return self.df

    # Method to load data from the Wikipedia link
    def _load_data(self):
        logger.info("Loading data from Wikipedia link...")
        url = "https://en.wikipedia.org/wiki/Iris_flower_data_set"
        iris_data = pd.read_html(url)[0]
        return iris_data

    # Method to save data
    def save(self, df: pd.DataFrame, out_path: str):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    iris_gen = IrisDataSetGenerator()
    iris_gen.create(save_path=TRAIN_PATH)
    iris_gen.create(save_path=INFERENCE_PATH)
    logger.info("Script completed successfully.")
