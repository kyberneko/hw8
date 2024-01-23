import argparse
import os
import json
import logging
import pickle
import sys
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import get_project_dir, configure_logging
import mlflow
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.getenv('CONF_PATH')

try:
    # Loads configuration settings from JSON
    with open("../" + CONF_FILE, "r") as file:
        conf = json.load(file)
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    sys.exit(1)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")


# PyTorch neural network model
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def main():
    try:
        configure_logging()

        logging.info(f"Loading data from {TRAIN_PATH}...")
        iris_data = pd.read_csv(TRAIN_PATH)
        logging.info("Encoding the target variable")
        le = LabelEncoder()
        iris_data['Species'] = le.fit_transform(iris_data['Species'])
        iris_data.head()

        logging.info("Splitting data into training and test sets...")
        X = iris_data.drop('Species', axis=1).values
        y = iris_data['Species'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        logging.info("Scaling data...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logging.info("Converting data to tensors")
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        # Step 5: Initialize model, loss function, and optimizer
        input_size = X_train.shape[1]
        output_size = len(torch.unique(y_train))
        model = SimpleModel(input_size, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create DataLoader for training
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        logging.info("Training the model")
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            logging.info("Evaluating the model")
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_test).sum().item() / len(y_test)
                logging.info(f"Test Accuracy: {accuracy:.2%}")

            logging.info("Saving the model...")
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)

            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
            with open(path, 'wb') as f:
                pickle.dump(model, f)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
