"""The code trains a simple neural network on an Iris dataset, evaluates its performance, and saves the trained model. """
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
from dotenv import load_dotenv


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def load_config(conf_path):
    try:
        with open("../" + conf_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at path: {conf_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


def load_data(train_path):
    try:
        logging.info(f"Loading data from {train_path}...")
        iris_data = pd.read_csv(train_path)
        logging.info("Encoding the target variable")
        le = LabelEncoder()
        iris_data['Species'] = le.fit_transform(iris_data['Species'])
        iris_data.head()
        return iris_data
    except FileNotFoundError:
        logging.error(f"Data file not found at path: {train_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"Data file is empty: {train_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)


def prepare_data(iris_data):
    try:
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

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        sys.exit(1)


def initialize_model(input_size, output_size):
    try:
        model = SimpleModel(input_size, output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model, criterion, optimizer
    except Exception as e:
        logging.error(f"Error initializing the model: {e}")
        sys.exit(1)


def create_dataloader(X_train, y_train, batch_size=32, shuffle=True):
    try:
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader
    except Exception as e:
        logging.error(f"Error creating DataLoader: {e}")
        sys.exit(1)


def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    try:
        logging.info("Training the model")
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        sys.exit(1)


def evaluate_model(model, X_test, y_test):
    try:
        logging.info("Evaluating the model")
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            logging.info(f"Test Accuracy: {accuracy:.2%}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        sys.exit(1)


def save_model(model, model_dir):
    logging.info("Saving the model...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save both the class definition and the model instance
    path = os.path.join(model_dir, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
    with open(path, 'wb') as f:
        pickle.dump(SimpleModel, f)  # Save the class definition
        pickle.dump(model.state_dict(), f)  # Save the model state dict


def main():
    try:
        configure_logging()
        load_dotenv()
        global conf
        conf = load_config(os.getenv('CONF_PATH'))
        data_dir = get_project_dir(conf['general']['data_dir'])
        model_dir = get_project_dir(conf['general']['models_dir'])
        train_path = os.path.join(data_dir, conf['train']['table_name'])

        iris_data = load_data(train_path)
        X_train, X_test, y_train, y_test = prepare_data(iris_data)
        input_size = X_train.shape[1]
        output_size = len(torch.unique(y_train))

        model, criterion, optimizer = initialize_model(input_size, output_size)
        train_loader = create_dataloader(X_train, y_train)
        train_model(model, train_loader, criterion, optimizer)
        evaluate_model(model, X_test, y_test)
        save_model(model, model_dir)

    except Exception as e:
        logging.error(f"The model was not saved...")
        sys.exit(1)


if __name__ == "__main__":
    main()
